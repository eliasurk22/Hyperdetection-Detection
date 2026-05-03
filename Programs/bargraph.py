import numpy as np
import matplotlib.pyplot as plt

# Metrics per fold (best values)
folds = ['Fold 0', 'Fold 1', 'Fold 2']

mlp_acc = [0.8684, 0.8979, 0.8547]
cnn_acc = [0.7469, 0.8055, 0.7709]

mlp_f1  = [0.8118, 0.7770, 0.8102]
cnn_f1  = [0.5320, 0.4414, 0.7069]

mlp_loss = [0.3125, 0.2530, 0.3458]
cnn_loss = [0.5138, 0.4272, 0.4584]

x = np.arange(len(folds))
width = 0.35

def plot_metric(mlp_vals, cnn_vals, ylabel, title, fname):
    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, mlp_vals, width, label='MLP')
    plt.bar(x + width/2, cnn_vals, width, label='1D-CNN')

    plt.xticks(x, folds)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

plot_metric(mlp_acc,  cnn_acc,  'Accuracy', 'Accuracy by Fold: MLP vs 1D-CNN',  'acc_bar.png')
plot_metric(mlp_f1,   cnn_f1,   'F1-score', 'F1-score by Fold: MLP vs 1D-CNN', 'f1_bar.png')
plot_metric(mlp_loss, cnn_loss, 'Loss',     'Loss by Fold: MLP vs 1D-CNN',     'loss_bar.png')
