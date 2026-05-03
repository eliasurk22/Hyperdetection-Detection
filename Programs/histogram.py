import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 1. ENTER YOUR RESULTS (per-fold values)
# ======================================================

# ---- MLP ----
mlp_acc  = [0.8684, 0.8979, 0.8547]
mlp_f1   = [0.8118, 0.7770, 0.8102]
mlp_loss = [0.3125, 0.2530, 0.3458]

# ---- 1D-CNN ----
cnn_acc  = [0.7469, 0.8055, 0.7709]
cnn_f1   = [0.5320, 0.4414, 0.7069]
cnn_loss = [0.5138, 0.4272, 0.4584]

# ======================================================
# 2. COMPUTE MEAN METRICS
# ======================================================
mlp_acc_mean  = np.mean(mlp_acc) * 100
cnn_acc_mean  = np.mean(cnn_acc) * 100

mlp_f1_mean   = np.mean(mlp_f1) * 100
cnn_f1_mean   = np.mean(cnn_f1) * 100

mlp_loss_mean = np.mean(mlp_loss)
cnn_loss_mean = np.mean(cnn_loss)

# For plotting
models = ['MLP', '1D-CNN']
x = np.arange(len(models))
bar_width = 0.15   # reduced width

# ======================================================
# 3. CREATE FIGURE
# ======================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Performance Comparison: MLP vs 1D-CNN", fontsize=14)

# Colors
colors = ['tab:blue', 'tab:orange']

# ---------- (a) Accuracy ----------
ax = axs[0, 0]
acc_vals = [mlp_acc_mean, cnn_acc_mean]
ax.bar(x, acc_vals, width=bar_width, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("a) Accuracy")

for i, v in enumerate(acc_vals):
    ax.text(i, v + 1, f"{v:.2f}", ha='center', fontsize=8)


# ---------- (b) Loss ----------
ax = axs[0, 1]
loss_vals = [mlp_loss_mean, cnn_loss_mean]
ax.bar(x, loss_vals, width=bar_width, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Loss")
ax.set_title("b) Loss")

for i, v in enumerate(loss_vals):
    ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=8)


# ---------- (c) F1-Score ----------
ax = axs[1, 0]
f1_vals = [mlp_f1_mean, cnn_f1_mean]
ax.bar(x, f1_vals, width=bar_width, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("F1-Score (%)")
ax.set_ylim(0, 100)
ax.set_title("c) F1-Score")

for i, v in enumerate(f1_vals):
    ax.text(i, v + 1, f"{v:.2f}", ha='center', fontsize=8)


# ---------- (d) Summary Box ----------
axs[1, 1].axis('off')
axs[1, 1].text(0.5, 0.5, "Model Comparison\nMLP vs 1D-CNN",
               ha='center', va='center', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
