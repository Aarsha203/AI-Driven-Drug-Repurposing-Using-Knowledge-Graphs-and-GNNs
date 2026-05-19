import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch_geometric.nn import GCNConv

try:
    import seaborn as sns
except ImportError:
    raise ImportError("Install seaborn using: pip install seaborn")

# Optional UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


BASE_DIR = Path(r"C:\Users\aar1\Downloads\hetionet_course_data")
KG_DIR = BASE_DIR / "kg_preprocessed"
RESULT_DIR = KG_DIR / "gnn_results"
PRED_DIR = KG_DIR / "repurposing_predictions"

DATA_PATH = KG_DIR / "hetionet_pyg_data.pt"
MODEL_PATH = RESULT_DIR / "best_model.pt"
NODE_MAP_PATH = KG_DIR / "node_to_idx.json"
EDGES_FILE = KG_DIR / "filtered_7_edge_types.tsv"
NODES_FILE = BASE_DIR / "hetionet-v1.0-nodes.tsv"
PRED_FILE = PRED_DIR / "all_unknown_drug_disease_predictions.csv"

OUTDIR = KG_DIR / "visualisation_report"
OUTDIR.mkdir(exist_ok=True)

for p in [DATA_PATH, MODEL_PATH, NODE_MAP_PATH, EDGES_FILE, NODES_FILE, PRED_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

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
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return self.link_head(torch.cat([src, dst], dim=1)).view(-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load(DATA_PATH, weights_only=False).to(device)
nodes = pd.read_csv(NODES_FILE, sep="\t")
edges = pd.read_csv(EDGES_FILE, sep="\t")
pred = pd.read_csv(PRED_FILE)

with open(NODE_MAP_PATH, "r") as f:
    node_to_idx = json.load(f)

idx_to_node = {v: k for k, v in node_to_idx.items()}

model = GCNLinkPredictor(data.x.shape[1], hidden_channels=64, dropout=0.3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==============================
# 1. Heatmap: top 10 drugs x top 5 diseases
# ==============================
top_drugs = (
    pred.groupby("drug_name")["prediction_score"]
    .max()
    .sort_values(ascending=False)
    .head(10)
    .index
)

top_diseases = (
    pred.groupby("disease_name")["prediction_score"]
    .max()
    .sort_values(ascending=False)
    .head(5)
    .index
)

heatmap_df = pred[
    pred["drug_name"].isin(top_drugs) &
    pred["disease_name"].isin(top_diseases)
]

heatmap_matrix = heatmap_df.pivot_table(
    index="drug_name",
    columns="disease_name",
    values="prediction_score",
    aggfunc="max"
)

plt.figure(figsize=(12, 7))
sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="viridis")
plt.title("Top Drug-Disease Repurposing Prediction Scores")
plt.xlabel("Disease")
plt.ylabel("Drug")
plt.tight_layout()
plt.savefig(OUTDIR / "top10_drugs_top5_diseases_heatmap.png", dpi=300)
plt.close()

# ==============================
# 2. 2-hop KG neighbourhood of top drug
# ==============================
top_drug = pred.iloc[0]["drug_name"]
top_drug_id = pred.iloc[0]["drug_id"]

G = nx.Graph()

for _, row in edges.iterrows():
    G.add_edge(row["source"], row["target"], edge_type=row["edge_type"])

neighbors_1 = set(G.neighbors(top_drug_id))
neighbors_2 = set()

for n in neighbors_1:
    neighbors_2.update(G.neighbors(n))

sub_nodes = {top_drug_id} | neighbors_1 | neighbors_2

# Limit for readable plot
sub_nodes = list(sub_nodes)[:120]
subG = G.subgraph(sub_nodes).copy()

id_to_name = dict(zip(nodes["id"], nodes["name"]))
id_to_kind = dict(zip(nodes["id"], nodes["kind"]))

labels = {n: id_to_name.get(n, n) for n in subG.nodes()}

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(subG, seed=42, k=0.35)

node_colors = []
for n in subG.nodes():
    kind = id_to_kind.get(n, "Unknown")
    if kind == "Compound":
        node_colors.append("skyblue")
    elif kind == "Disease":
        node_colors.append("salmon")
    elif kind == "Gene":
        node_colors.append("lightgreen")
    else:
        node_colors.append("lightgray")

nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=350)
nx.draw_networkx_edges(subG, pos, alpha=0.4)
nx.draw_networkx_labels(subG, pos, labels=labels, font_size=6)

plt.title(f"2-hop Knowledge Graph Neighbourhood of Top Drug: {top_drug}")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTDIR / "top_drug_2hop_kg_neighbourhood.png", dpi=300)
plt.close()

# ==============================
# 3. GNN embeddings + t-SNE / UMAP
# ==============================
with torch.no_grad():
    embeddings = model.encode(data.x, data.edge_index).cpu().numpy()

embedding_df = pd.DataFrame(embeddings)
embedding_df["node_id"] = [idx_to_node[i] for i in range(len(embeddings))]
embedding_df = embedding_df.merge(
    nodes[["id", "name", "kind"]],
    left_on="node_id",
    right_on="id",
    how="left"
)

# Sample for speed/readability
sample_df = embedding_df.sample(
    n=min(5000, len(embedding_df)),
    random_state=42
)

X = sample_df.iloc[:, :embeddings.shape[1]].values

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(X)

sample_df["TSNE1"] = tsne_result[:, 0]
sample_df["TSNE2"] = tsne_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=sample_df,
    x="TSNE1",
    y="TSNE2",
    hue="kind",
    s=12,
    alpha=0.75
)
plt.title("t-SNE of GNN Node Embeddings by Node Type")
plt.tight_layout()
plt.savefig(OUTDIR / "gnn_node_embeddings_tsne.png", dpi=300)
plt.close()

if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(X)

    sample_df["UMAP1"] = umap_result[:, 0]
    sample_df["UMAP2"] = umap_result[:, 1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=sample_df,
        x="UMAP1",
        y="UMAP2",
        hue="kind",
        s=12,
        alpha=0.75
    )
    plt.title("UMAP of GNN Node Embeddings by Node Type")
    plt.tight_layout()
    plt.savefig(OUTDIR / "gnn_node_embeddings_umap.png", dpi=300)
    plt.close()

sample_df.to_csv(OUTDIR / "embedding_visualisation_coordinates.csv", index=False)

# ==============================
# Final report
# ==============================
report = OUTDIR / "final_visualisation_report.txt"

with open(report, "w", encoding="utf-8") as f:
    f.write("Hetionet GNN Visualisation and Final Report\n")
    f.write("==========================================\n\n")
    f.write(f"Top predicted drug: {top_drug}\n")
    f.write(f"Top drug ID: {top_drug_id}\n\n")
    f.write("Generated figures:\n")
    f.write("1. top10_drugs_top5_diseases_heatmap.png\n")
    f.write("2. top_drug_2hop_kg_neighbourhood.png\n")
    f.write("3. gnn_node_embeddings_tsne.png\n")
    if HAS_UMAP:
        f.write("4. gnn_node_embeddings_umap.png\n")
    f.write("\nInterpretation:\n")
    f.write(
        "The heatmap summarizes high-scoring drug-disease repurposing predictions. "
        "The 2-hop network shows the local biological neighbourhood of the top-ranked drug. "
        "The t-SNE/UMAP plots visualize whether GNN embeddings separate biologically distinct node types."
    )

print("Visualisation complete.")
print("Outputs saved in:", OUTDIR)