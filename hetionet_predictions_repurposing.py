import json
from pathlib import Path
from urllib.parse import quote
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# ==============================
# Paths
# ==============================
BASE_DIR = Path(r"C:\Users\aar1\Downloads\hetionet_course_data")
KG_DIR = BASE_DIR / "kg_preprocessed"
RESULT_DIR = KG_DIR / "gnn_results"

DATA_PATH = KG_DIR / "hetionet_pyg_data.pt"
MODEL_PATH = RESULT_DIR / "best_model.pt"
NODE_MAP_PATH = KG_DIR / "node_to_idx.json"
NODES_FILE = BASE_DIR / "hetionet-v1.0-nodes.tsv"
POSITIVE_FILE = KG_DIR / "positive_drug_disease_pairs.tsv"

OUTDIR = KG_DIR / "repurposing_predictions"
OUTDIR.mkdir(exist_ok=True)

# Change disease here
FOCUS_DISEASE_NAME = "Alzheimer's disease"

HIGH_CONF_THRESHOLD = 0.70

# ==============================
# Model class
# ==============================
class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.link_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        pair_emb = torch.cat([src, dst], dim=1)
        return self.link_head(pair_emb).view(-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

# ==============================
# PubMed search helper
# ==============================
def pubmed_count(query):
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={quote(query)}&retmode=xml"
    )

    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        count = root.findtext("Count")
        return int(count)

    except Exception:
        return None

# ==============================
# Load files
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load(DATA_PATH, weights_only=False).to(device)

nodes = pd.read_csv(NODES_FILE, sep="\t")

with open(NODE_MAP_PATH, "r") as f:
    node_to_idx = json.load(f)

idx_to_node = {v: k for k, v in node_to_idx.items()}

positive_pairs = pd.read_csv(POSITIVE_FILE, sep="\t")
known_positive_set = set(zip(positive_pairs["source"], positive_pairs["target"]))

# ==============================
# Load trained model
# ==============================
model = GCNLinkPredictor(
    in_channels=data.x.shape[1],
    hidden_channels=64,
    dropout=0.3
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==============================
# Identify compounds and diseases
# ==============================
nodes_in_kg = nodes[nodes["id"].isin(node_to_idx.keys())].copy()

compound_df = nodes_in_kg[nodes_in_kg["kind"] == "Compound"].copy()
disease_df = nodes_in_kg[nodes_in_kg["kind"] == "Disease"].copy()

print("Compounds:", len(compound_df))
print("Diseases:", len(disease_df))

# ==============================
# Score all unknown drug-disease pairs
# ==============================
print("\nEncoding graph...")
with torch.no_grad():
    z = model.encode(data.x, data.edge_index)

all_predictions = []

print("\nScoring unknown drug-disease pairs...")

for _, drug_row in compound_df.iterrows():
    drug_id = drug_row["id"]
    drug_name = drug_row["name"]

    for _, disease_row in disease_df.iterrows():
        disease_id = disease_row["id"]
        disease_name = disease_row["name"]

        if (drug_id, disease_id) in known_positive_set:
            continue

        src_idx = node_to_idx[drug_id]
        dst_idx = node_to_idx[disease_id]

        edge_label_index = torch.tensor(
            [[src_idx], [dst_idx]],
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():
            score = model.decode(z, edge_label_index).item()

        all_predictions.append([
            drug_id,
            drug_name,
            disease_id,
            disease_name,
            score
        ])

pred_df = pd.DataFrame(
    all_predictions,
    columns=[
        "drug_id",
        "drug_name",
        "disease_id",
        "disease_name",
        "prediction_score"
    ]
)

pred_df = pred_df.sort_values("prediction_score", ascending=False)

pred_df.to_csv(
    OUTDIR / "all_unknown_drug_disease_predictions.csv",
    index=False
)

high_conf = pred_df[pred_df["prediction_score"] >= HIGH_CONF_THRESHOLD]

high_conf.to_csv(
    OUTDIR / "high_confidence_repurposing_hits_score_gt_0.7.csv",
    index=False
)

print("\nTotal unknown predictions:", len(pred_df))
print("High-confidence hits:", len(high_conf))

# ==============================
# Focus on one disease
# ==============================
focus_hits = pred_df[
    pred_df["disease_name"].str.lower() == FOCUS_DISEASE_NAME.lower()
].copy()

focus_hits = focus_hits.sort_values("prediction_score", ascending=False)

focus_hits.to_csv(
    OUTDIR / f"top_hits_for_{FOCUS_DISEASE_NAME.replace(' ', '_')}.csv",
    index=False
)

print(f"\nTop hits for {FOCUS_DISEASE_NAME}:")
print(focus_hits.head(20))

# ==============================
# PubMed evidence for top hits
# ==============================
top_pubmed_hits = focus_hits.head(20).copy()

pubmed_counts = []

print("\nSearching PubMed evidence for top disease-specific hits...")

for _, row in top_pubmed_hits.iterrows():
    query = f'"{row["drug_name"]}" AND "{row["disease_name"]}"'
    count = pubmed_count(query)
    pubmed_counts.append(count)

top_pubmed_hits["pubmed_query"] = [
    f'"{row["drug_name"]}" AND "{row["disease_name"]}"'
    for _, row in top_pubmed_hits.iterrows()
]

top_pubmed_hits["pubmed_result_count"] = pubmed_counts

top_pubmed_hits.to_csv(
    OUTDIR / f"pubmed_evidence_top_hits_{FOCUS_DISEASE_NAME.replace(' ', '_')}.csv",
    index=False
)

# ==============================
# Cross-check known repurposed drugs
# ==============================
known_repurposed = ["Metformin", "Thalidomide", "Aspirin"]

crosscheck_rows = []

for drug in known_repurposed:
    temp = pred_df[
        pred_df["drug_name"].str.lower() == drug.lower()
    ].copy()

    if len(temp) == 0:
        crosscheck_rows.append([drug, "Not found", None, None, None])
        continue

    temp = temp.sort_values("prediction_score", ascending=False)
    best_row = temp.iloc[0]

    crosscheck_rows.append([
        drug,
        best_row["disease_name"],
        best_row["prediction_score"],
        temp["prediction_score"].rank(ascending=False).iloc[0],
        len(temp)
    ])

crosscheck_df = pd.DataFrame(
    crosscheck_rows,
    columns=[
        "known_repurposed_drug",
        "top_predicted_disease",
        "top_prediction_score",
        "rank",
        "total_disease_predictions"
    ]
)

crosscheck_df.to_csv(
    OUTDIR / "known_repurposed_drug_crosscheck.csv",
    index=False
)

print("\nKnown repurposed drug cross-check:")
print(crosscheck_df)

# ==============================
# Final summary
# ==============================
summary_file = OUTDIR / "repurposing_prediction_summary.txt"

with open(summary_file, "w", encoding="utf-8") as f:
    f.write("Drug Repurposing Prediction Summary\n")
    f.write("==================================\n\n")
    f.write(f"Total unknown drug-disease pairs scored: {len(pred_df)}\n")
    f.write(f"High-confidence hits score >= {HIGH_CONF_THRESHOLD}: {len(high_conf)}\n")
    f.write(f"Focus disease: {FOCUS_DISEASE_NAME}\n\n")
    f.write("Top 10 disease-specific hits:\n")
    f.write(focus_hits.head(10).to_string(index=False))
    f.write("\n\nKnown repurposed drug cross-check:\n")
    f.write(crosscheck_df.to_string(index=False))

print("\nAll prediction outputs saved in:")
print(OUTDIR)