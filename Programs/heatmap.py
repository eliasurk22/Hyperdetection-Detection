import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------ 1. Data ------------

# best metrics by hand from logs
mlp_acc = [0.8684, 0.8979, 0.8547]
mlp_f1  = [0.8118, 0.7770, 0.8102]

cnn_acc = [0.7469, 0.8055, 0.7709]
cnn_f1  = [0.5320, 0.4414, 0.7069]

fold_labels = ['Fold 0', 'Fold 1', 'Fold 2']

# ------------ 2. Comparison heat map (accuracy) ------------

comp_acc = pd.DataFrame(
    [mlp_acc, cnn_acc],
    index=['MLP', '1D-CNN'],
    columns=fold_labels
)

plt.figure(figsize=(6, 3))
sns.heatmap(comp_acc, annot=True, fmt=".3f", cmap="Blues")
plt.title("Accuracy by Fold: MLP vs 1D-CNN")
plt.ylabel("Model")
plt.xlabel("Fold")
plt.tight_layout()
plt.show()

# If you prefer F1 comparison instead:
comp_f1 = pd.DataFrame(
    [mlp_f1, cnn_f1],
    index=['MLP', '1D-CNN'],
    columns=fold_labels
)

plt.figure(figsize=(6, 3))
sns.heatmap(comp_f1, annot=True, fmt=".3f", cmap="Greens")
plt.title("F1-score by Fold: MLP vs 1D-CNN")
plt.ylabel("Model")
plt.xlabel("Fold")
plt.tight_layout()
plt.show()

# ------------ 3. Separate heat map for MLP ------------

mlp_df = pd.DataFrame(
    {'Accuracy': mlp_acc, 'F1-score': mlp_f1},
    index=fold_labels
).T  # rows=metrics, cols=folds

plt.figure(figsize=(6, 3))
sns.heatmap(mlp_df, annot=True, fmt=".3f", cmap="Oranges")
plt.title("MLP Performance by Fold")
plt.ylabel("Metric")
plt.xlabel("Fold")
plt.tight_layout()
plt.show()

# ------------ 4. Separate heat map for 1D-CNN ------------

cnn_df = pd.DataFrame(
    {'Accuracy': cnn_acc, 'F1-score': cnn_f1},
    index=fold_labels
).T

plt.figure(figsize=(6, 3))
sns.heatmap(cnn_df, annot=True, fmt=".3f", cmap="Purples")
plt.title("1D-CNN Performance by Fold")
plt.ylabel("Metric")
plt.xlabel("Fold")
plt.tight_layout()
plt.show()
