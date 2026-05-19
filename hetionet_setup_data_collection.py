import os
import gzip
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
import networkx as nx

# ==============================
# Output folder
# ==============================
OUTDIR = Path(r"C:\Users\aar1\Downloads\hetionet_course_data")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Hetionet download URLs
# ==============================
edges_url = "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
nodes_url = "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-nodes.tsv"

edges_gz = OUTDIR / "hetionet-v1.0-edges.sif.gz"
edges_file = OUTDIR / "hetionet-v1.0-edges.sif"
nodes_file = OUTDIR / "hetionet-v1.0-nodes.tsv"

# ==============================
# Download files
# ==============================
def download_file(url, output_path):
    if output_path.exists():
        print(f"Already exists: {output_path}")
    else:
        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved: {output_path}")

download_file(edges_url, edges_gz)
download_file(nodes_url, nodes_file)

# Unzip edges file
if not edges_file.exists():
    print("Extracting edges.sif.gz...")
    with gzip.open(edges_gz, "rb") as f_in:
        with open(edges_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

# ==============================
# Load data
# ==============================
print("\nLoading Hetionet files...")

edges = pd.read_csv(
    edges_file,
    sep="\t",
    header=None,
    names=["source", "edge_type", "target"]
)

nodes = pd.read_csv(nodes_file, sep="\t")

print("\nEdges file preview:")
print(edges.head())

print("\nNodes file preview:")
print(nodes.head())

# ==============================
# Explore node types
# ==============================
print("\nNode type counts:")
node_type_counts = nodes["kind"].value_counts()
print(node_type_counts)

node_type_counts.to_csv(OUTDIR / "node_type_counts.csv")

# ==============================
# Explore edge types
# ==============================
print("\nEdge type counts:")
edge_type_counts = edges["edge_type"].value_counts()
print(edge_type_counts)

edge_type_counts.to_csv(OUTDIR / "edge_type_counts.csv")

# ==============================
# Count drug-disease links
# ==============================
drug_disease_edges = edges[
    edges["edge_type"].str.contains("treats|palliates", case=False, na=False)
]

print("\nDrug-disease links:")
print(drug_disease_edges.head())
print(f"\nTotal drug-disease links: {len(drug_disease_edges)}")

drug_disease_edges.to_csv(
    OUTDIR / "drug_disease_links.csv",
    index=False
)

# ==============================
# Build NetworkX graph
# ==============================
print("\nBuilding NetworkX graph...")

G = nx.MultiDiGraph()

for _, row in nodes.iterrows():
    G.add_node(
        row["id"],
        name=row.get("name", ""),
        kind=row.get("kind", "")
    )

for _, row in edges.iterrows():
    G.add_edge(
        row["source"],
        row["target"],
        edge_type=row["edge_type"]
    )

print("\nGraph summary:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# ==============================
# Save summary report
# ==============================
summary_file = OUTDIR / "hetionet_summary_report.txt"

with open(summary_file, "w", encoding="utf-8") as f:
    f.write("Hetionet Setup & Data Collection Summary\n")
    f.write("=======================================\n\n")
    f.write(f"Total nodes: {G.number_of_nodes()}\n")
    f.write(f"Total edges: {G.number_of_edges()}\n")
    f.write(f"Total drug-disease links: {len(drug_disease_edges)}\n\n")

    f.write("Node type counts:\n")
    f.write(node_type_counts.to_string())
    f.write("\n\nEdge type counts:\n")
    f.write(edge_type_counts.to_string())

print("\nAnalysis complete.")
print(f"All files saved in: {OUTDIR}")
print(f"Summary report: {summary_file}")