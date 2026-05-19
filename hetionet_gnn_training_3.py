import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.nn import GCNConv

# ==============================
# Paths
# ==============================
BASE_DIR = Path(r"C:\Users\aar1\Downloads\hetionet_course_data\kg_preprocessed")
DATA_PATH = BASE_DIR / "hetionet_pyg_data.pt"

OUTDIR = BASE_DIR / "gnn_results"
OUTDIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = OUTDIR / "best_model.pt"

# ==============================
# Reproducibility
# ==============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# Load PyG Data
# ==============================
data = torch.load(DATA_PATH, weights_only=False)
data = data.to(device)

print(data)

# ==============================
# Train/Val/Test split: 70/15/15
# ==============================
num_pairs = data.edge_label.shape[0]
indices = torch.randperm(num_pairs)

train_end = int(0.70 * num_pairs)
val_end = int(0.85 * num_pairs)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

train_edge_label_index = data.edge_label_index[:, train_idx]
train_labels = data.edge_label[train_idx].float()

val_edge_label_index = data.edge_label_index[:, val_idx]
val_labels = data.edge_label[val_idx].float()

test_edge_label_index = data.edge_label_index[:, test_idx]
test_labels = data.edge_label[test_idx].float()

print("Train pairs:", train_labels.shape[0])
print("Validation pairs:", val_labels.shape[0])
print("Test pairs:", test_labels.shape[0])

# ==============================
# 2-layer GCN Link Prediction Model
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

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

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


model = GCNLinkPredictor(
    in_channels=data.x.shape[1],
    hidden_channels=64,
    dropout=0.3
).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ==============================
# Evaluation function
# ==============================
@torch.no_grad()
def evaluate(edge_label_index, labels):
    model.eval()

    probs = model(data.x, data.edge_index, edge_label_index)
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    auroc = roc_auc_score(labels_np, probs_np)
    auprc = average_precision_score(labels_np, probs_np)

    loss = criterion(probs, labels).item()

    return loss, auroc, auprc, probs_np, labels_np

# ==============================
# Training
# ==============================
epochs = 100

train_losses = []
val_losses = []
val_aurocs = []
val_auprcs = []

best_val_auroc = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    train_probs = model(data.x, data.edge_index, train_edge_label_index)
    train_loss = criterion(train_probs, train_labels)

    train_loss.backward()
    optimizer.step()

    val_loss, val_auroc, val_auprc, _, _ = evaluate(
        val_edge_label_index,
        val_labels
    )

    train_losses.append(train_loss.item())
    val_losses.append(val_loss)
    val_aurocs.append(val_auroc)
    val_auprcs.append(val_auprc)

    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | "
            f"Val AUPRC: {val_auprc:.4f}"
        )

print("\nBest validation AUROC:", best_val_auroc)
print("Best model saved:", BEST_MODEL_PATH)

# ==============================
# Training curves
# ==============================
plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "training_curves.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(val_aurocs, label="Validation AUROC")
plt.plot(val_auprcs, label="Validation AUPRC")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation AUROC and AUPRC")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "validation_metrics.png", dpi=300)
plt.close()

# ==============================
# Test evaluation
# ==============================
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

test_loss, test_auroc, test_auprc, test_probs, test_true = evaluate(
    test_edge_label_index,
    test_labels
)

print("\nTest Evaluation")
print("Test Loss:", round(test_loss, 4))
print("Test AUROC:", round(test_auroc, 4))
print("Test AUPRC:", round(test_auprc, 4))

# ==============================
# Confusion matrix
# ==============================
threshold = 0.5
test_pred = (test_probs >= threshold).astype(int)

cm = confusion_matrix(test_true, test_pred)

print("\nConfusion Matrix:")
print(cm)

np.savetxt(
    OUTDIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(OUTDIR / "confusion_matrix.png", dpi=300)
plt.close()

# ==============================
# ROC curve
# ==============================
fpr, tpr, _ = roc_curve(test_true, test_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUROC = {test_auroc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "roc_curve.png", dpi=300)
plt.close()

# ==============================
# Precision-Recall curve
# ==============================
precision, recall, _ = precision_recall_curve(test_true, test_probs)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"AUPRC = {test_auprc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "pr_curve.png", dpi=300)
plt.close()

# ==============================
# Save test predictions
# ==============================
results = np.column_stack([test_true, test_probs, test_pred])

np.savetxt(
    OUTDIR / "test_predictions.csv",
    results,
    delimiter=",",
    header="true_label,predicted_probability,predicted_label",
    comments=""
)

# ==============================
# Save metrics summary
# ==============================
with open(OUTDIR / "test_metrics_summary.txt", "w") as f:
    f.write("GNN Link Prediction Test Metrics\n")
    f.write("================================\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test AUROC: {test_auroc:.4f}\n")
    f.write(f"Test AUPRC: {test_auprc:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("\nAll outputs saved in:")
print(OUTDIR)