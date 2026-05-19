# AI-Driven-Drug-Repurposing-Using-Knowledge-Graphs-and-GNNs
AI-driven drug repurposing framework leveraging biomedical knowledge graphs and Graph Neural Networks (GNNs) for novel drug–disease link prediction, therapeutic candidate prioritization, and network-based validation using Hetionet, PyTorch Geometric, and NetworkX.
This repository presents a complete end-to-end computational drug repurposing pipeline based on biomedical knowledge graphs and Graph Neural Networks (GNNs). The workflow integrates heterogeneous biomedical relationships from Hetionet and applies graph representation learning for link prediction to identify potential new therapeutic indications for existing drugs.

Unlike traditional structure-based approaches, this framework leverages biological relationships among drugs, diseases, genes, pathways, and other biomedical entities to discover novel drug–disease associations.
_**Objectives**_
*Construct a biomedical knowledge graph from Hetionet
*Generate node features and graph representations
*Train a Graph Neural Network for drug–disease link prediction
*Predict novel therapeutic indications for approved drugs
*Validate predictions using literature evidence
*Visualize graph structures and learned embeddings

_**Workflow**_
**Phase 1 — Data Collection**
-Download Hetionet node and edge files
-Explore graph statistics
-Characterize node and edge distributions
**Phase 2 — Knowledge Graph Construction**
-Filter biologically relevant edge types
-Create node index mappings
-Generate positive and negative drug–disease pairs
-Build PyTorch Geometric graph objects
**Phase 3 — GNN Training**
-Two-layer Graph Convolutional Network (GCN)
-Link prediction framework
-Binary cross-entropy loss
-Adam optimizer
-AUROC and AUPRC evaluation
**Phase 4 — Drug Repurposing Prediction**
-Score unknown drug–disease pairs
-Rank high-confidence predictions
-Literature-based validation using PubMed
**Phase 5 — Visualization and Reporting**
-Knowledge graph visualization
-Drug–disease interaction heatmaps
-Embedding visualization (t-SNE / UMAP)
-Final report generation
