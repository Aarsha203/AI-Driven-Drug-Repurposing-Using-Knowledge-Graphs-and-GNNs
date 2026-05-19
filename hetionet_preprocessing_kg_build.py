import json
import random
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data

# ==============================
# Paths
# ==============================
BASE_DIR = Path(r"C:\Users\aar1\Downloads\hetionet_course_data")

edges_file = BASE_DIR / "hetionet-v1.0-edges.sif"
nodes_file = BASE_DIR / "hetionet-v1.0-nodes.tsv"

OUTDIR = BASE_DIR / "kg_preprocessed"
OUTDIR.mkdir(exist_ok=True)

# ==============================
# Load data
# ==============================
edges = pd.read_csv(
    edges_file,
    sep="\t",
    header=None,
    names=["source", "edge_type", "target"]
)

nodes = pd.read_csv(nodes_file, sep="\t")

print("Loaded data")
print("Nodes:", nodes.shape)
print("Edges:", edges.shape)

print("\nAvailable edge types:")
print(edges["edge_type"].value_counts())

# ==============================
# Filter biologically relevant edge types
# ==============================
# Hetionet edge codes:
# CtD = Compound treats Disease
# CpD = Compound palliates Disease
# CbG = Compound binds Gene
# CdG = Compound downregulates Gene
# CuG = Compound upregulates Gene
# GpBP = Gene participates Biological Process
# DaG = Disease associates Gene

selected_edge_types = [
    "CtD",
    "CpD",
    "CbG",
    "CdG",
    "CuG",
    "GpBP",
    "DaG"
]

filtered_edges = edges[edges["edge_type"].isin(selected_edge_types)].copy()

print("\nFiltered edges:")
print(filtered_edges["edge_type"].value_counts())
print("Total filtered edges:", len(filtered_edges))

filtered_edges.to_csv(
    OUTDIR / "filtered_7_edge_types.tsv",
    sep="\t",
    index=False
)

# ==============================
# Node to integer index mapping
# ==============================
all_nodes = pd.concat(
    [filtered_edges["source"], filtered_edges["target"]]
).unique()

node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
idx_to_node = {idx: node for node, idx in node_to_idx.items()}

with open(OUTDIR / "node_to_idx.json", "w") as f:
    json.dump(node_to_idx, f, indent=4)

with open(OUTDIR / "idx_to_node.json", "w") as f:
    json.dump(idx_to_node, f, indent=4)

print("\nTotal KG nodes after filtering:", len(node_to_idx))

# ==============================
# Create edge index
# ==============================
edge_source = filtered_edges["source"].map(node_to_idx)
edge_target = filtered_edges["target"].map(node_to_idx)

edge_index = torch.tensor(
    [edge_source.values, edge_target.values],
    dtype=torch.long
)

# ==============================
# One-hot encode node types
# ==============================
nodes_subset = nodes[nodes["id"].isin(node_to_idx.keys())].copy()
nodes_subset["node_idx"] = nodes_subset["id"].map(node_to_idx)
nodes_subset = nodes_subset.sort_values("node_idx")

encoder = OneHotEncoder(sparse_output=False)

node_type_features = encoder.fit_transform(
    nodes_subset[["kind"]]
)

x = torch.tensor(node_type_features, dtype=torch.float)

print("\nNode feature matrix shape:", x.shape)

node_type_df = pd.DataFrame(
    node_type_features,
    columns=encoder.get_feature_names_out(["kind"])
)

node_type_df.insert(0, "node_id", nodes_subset["id"].values)
node_type_df.insert(1, "node_idx", nodes_subset["node_idx"].values)
node_type_df.to_csv(OUTDIR / "node_type_onehot_features.csv", index=False)

# ==============================
# Positive drug-disease pairs
# ==============================
positive_edges = filtered_edges[
    filtered_edges["edge_type"].isin(["CtD", "CpD"])
].copy()

positive_pairs = positive_edges[["source", "target"]].drop_duplicates()

print("\nPositive drug-disease pairs:", len(positive_pairs))

positive_pairs.to_csv(
    OUTDIR / "positive_drug_disease_pairs.tsv",
    sep="\t",
    index=False
)

# ==============================
# Negative random drug-disease pairs
# ==============================
compound_nodes = nodes[
    (nodes["kind"] == "Compound") & (nodes["id"].isin(node_to_idx.keys()))
]["id"].tolist()

disease_nodes = nodes[
    (nodes["kind"] == "Disease") & (nodes["id"].isin(node_to_idx.keys()))
]["id"].tolist()

positive_set = set(
    zip(positive_pairs["source"], positive_pairs["target"])
)

negative_pairs = set()

random.seed(42)

while len(negative_pairs) < len(positive_pairs):
    drug = random.choice(compound_nodes)
    disease = random.choice(disease_nodes)

    if (drug, disease) not in positive_set:
        negative_pairs.add((drug, disease))

negative_pairs = pd.DataFrame(
    list(negative_pairs),
    columns=["source", "target"]
)

print("Negative random pairs:", len(negative_pairs))

negative_pairs.to_csv(
    OUTDIR / "negative_drug_disease_pairs.tsv",
    sep="\t",
    index=False
)

# ==============================
# Build labels for link prediction
# ==============================
pos_edge_label_index = torch.tensor(
    [
        positive_pairs["source"].map(node_to_idx).values,
        positive_pairs["target"].map(node_to_idx).values
    ],
    dtype=torch.long
)

neg_edge_label_index = torch.tensor(
    [
        negative_pairs["source"].map(node_to_idx).values,
        negative_pairs["target"].map(node_to_idx).values
    ],
    dtype=torch.long
)

edge_label_index = torch.cat(
    [pos_edge_label_index, neg_edge_label_index],
    dim=1
)

edge_label = torch.cat(
    [
        torch.ones(pos_edge_label_index.shape[1]),
        torch.zeros(neg_edge_label_index.shape[1])
    ]
).long()

# ==============================
# PyTorch Geometric Data object
# ==============================
data = Data(
    x=x,
    edge_index=edge_index,
    edge_label_index=edge_label_index,
    edge_label=edge_label
)

torch.save(data, OUTDIR / "hetionet_pyg_data.pt")

print("\nPyTorch Geometric Data object:")
print(data)

print("\nSaved files in:")
print(OUTDIR)