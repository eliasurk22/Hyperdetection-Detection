import matplotlib.pyplot as plt

folds = [0, 1, 2]

mlp_acc = [0.8684, 0.8979, 0.8547]
mlp_f1  = [0.8118, 0.7770, 0.8102]

cnn_acc = [0.7469, 0.8055, 0.7709]
cnn_f1  = [0.5320, 0.4414, 0.7069]

# Accuracy line graph
plt.figure(figsize=(6,4))
plt.plot(folds, mlp_acc, marker='o', label='MLP Accuracy')
plt.plot(folds, cnn_acc, marker='s', label='CNN Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0.4, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('acc_mlp_cnn.png', dpi=300)

# F1-score line graph
plt.figure(figsize=(6,4))
plt.plot(folds, mlp_f1, marker='o', label='MLP F1-score')
plt.plot(folds, cnn_f1, marker='s', label='CNN F1-score')
plt.xlabel('Fold')
plt.ylabel('F1-score')
plt.ylim(0.3, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('f1_mlp_cnn.png', dpi=300)
