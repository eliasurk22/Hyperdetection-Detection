import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

base = r'C:\Users\Elias John Sabu\Downloads\8th Sem Project\uci2_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

class CNN1D(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        reduced_len = seq_len // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced_len, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


epochs = 10
batch_size = 128
lr = 1e-3

for fold in [0, 1, 2]:
    csv_path = os.path.join(base, f'uci2_signal_fold_{fold}_full.csv')
    if not os.path.exists(csv_path):
        print(f"Fold {fold}: file not found, skipping")
        continue

    print(f"\n=== Fold {fold} ===")
    df = pd.read_csv(csv_path)

    ppg_cols = [c for c in df.columns if c.startswith('ppg_')]
    X = df[ppg_cols].values.astype(np.float32)              # (N, 625)
    y = ((df['SP'] >= 140) | (df['DP'] >= 90)).astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train).unsqueeze(1)  # (N,1,L)
    X_val_t   = torch.tensor(X_val).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size)

    model = CNN1D(seq_len=X_train.shape[1]).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        tot_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * xb.size(0)
        train_loss = tot_loss / len(train_loader.dataset)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                p = out.argmax(dim=1).cpu().numpy()
                preds.extend(p)
                labels.extend(yb.numpy())
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f"Fold {fold} Epoch {ep:02d} | loss={train_loss:.4f} | acc={acc:.4f} | f1={f1:.4f}")
