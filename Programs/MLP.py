import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

base = r'C:\Users\Elias John Sabu\Downloads\8th Sem Project\uci2_dataset'
X_folder = os.path.join(base, 'preproc_feats')
batch_size = 128
epochs = 10
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

for fold in [0, 1, 2]:
    X_path = os.path.join(X_folder, f'X_fold_{fold}_with_pattern.csv')
    y_path = os.path.join(X_folder, f'y_fold_{fold}.csv')

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print(f"Missing data for fold {fold}, skipping")
        continue

    print(f"\n=== Training MLP on fold {fold} ===")
    X = pd.read_csv(X_path).values
    y = pd.read_csv(y_path)['HTN'].values  # 0/1 label

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size)

    model = MLP(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc, best_f1 = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        best_acc = max(best_acc, acc)
        best_f1 = max(best_f1, f1)
        print(f"Fold {fold} | Epoch {epoch:02d} | loss={avg_train_loss:.4f} | val_acc={acc:.4f} | val_f1={f1:.4f}")

    print(f"Fold {fold} BEST: acc={best_acc:.4f}, f1={best_f1:.4f}")
