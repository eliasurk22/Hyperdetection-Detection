import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

base = r'C:\Users\Elias John Sabu\Downloads\8th Sem Project\uci2_dataset'

# ==== Training config ====
max_samples_per_fold = 20000   # subsample for speed; increase if your PC allows
epochs = 10                    # a bit more training
batch_size = 128
lr = 1e-3

# ---- transformer hyperparameters (efficient but non-trivial) ----
d_model = 64        # embedding size
n_heads = 4
n_layers = 2
ff_dim = 128        # feedforward dim


class PPGTransformer(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, n_layers, ff_dim, num_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x = self.input_proj(x)     # (batch, seq_len, d_model)
        x = self.encoder(x)        # (batch, seq_len, d_model)
        x = x.mean(dim=1)          # global average pooling over time
        x = self.classifier(x)     # (batch, num_classes)
        return x


for fold in [0, 1, 2]:
    csv_path = os.path.join(base, f'uci2_signal_fold_{fold}_full.csv')
    if not os.path.exists(csv_path):
        print(f"\nFold {fold}: file not found, skipping")
        continue

    print(f"\n=== Transformer on fold {fold} ===")
    df = pd.read_csv(csv_path)

    # ---- build HTN labels ----
    ppg_cols = [c for c in df.columns if c.startswith('ppg_')]
    X_all = df[ppg_cols].values.astype(np.float32)   # (N, 625)
    y_all = ((df['SP'] >= 140) | (df['DP'] >= 90)).astype(int).values

    # ---- stratified subsample for speed ----
    if len(df) > max_samples_per_fold:
        X_used, _, y_used, _ = train_test_split(
            X_all, y_all,
            train_size=max_samples_per_fold,
            stratify=y_all,
            random_state=42
        )
        print(f"Using subsample of {max_samples_per_fold} / {len(df)} samples")
    else:
        X_used, y_used = X_all, y_all
        print(f"Using all {len(df)} samples")

    # ---- train/val split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_used, y_used,
        test_size=0.2,
        random_state=42,
        stratify=y_used
    )

    # tensors: (batch, seq_len, 1)
    X_train_t = torch.tensor(X_train).unsqueeze(-1)
    X_val_t   = torch.tensor(X_val).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size)

    seq_len = X_train.shape[1]
    model = PPGTransformer(seq_len, d_model, n_heads, n_layers, ff_dim).to(device)

    # ---- class-weighted loss to handle imbalance ----
    class_counts = np.bincount(y_train)
    class_weights = torch.tensor(
        [1.0 / class_counts[0], 1.0 / class_counts[1]],
        dtype=torch.float32,
        device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        train_loss = total_loss / len(train_loader.dataset)

        # validation
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
        print(f"Fold {fold} Epoch {epoch:02d} | loss={train_loss:.4f} | "
              f"val_acc={acc:.4f} | val_f1={f1:.4f}")
