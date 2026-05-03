import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

base = r'C://Users//Elias John Sabu//Downloads//8th Sem Project//uci2_dataset'  # CHANGE if needed
out_folder = os.path.join(base, 'preproc_feats')
os.makedirs(out_folder, exist_ok=True)

# standard scaler per fold (you can also fit on fold_0 and apply to others if you want)
for fold in [0, 1, 2]:
    fname = f'feat_fold_{fold}.csv'
    fpath = os.path.join(base, fname)
    if not os.path.exists(fpath):
        print(f"Missing {fpath}, skipping")
        continue

    print(f"\nProcessing {fname}")
    df = pd.read_csv(fpath)

    # 1) Drop ID columns
    drop_cols = [c for c in ['patient', 'trial'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 2) Handle missing values (fill NaNs with column mean)
    df = df.fillna(df.mean(numeric_only=True))

    # 3) Create labels
    # Regression targets
    y_sp = df['SP'].values
    y_dp = df['DP'].values

    # Classification target: hypertension if SP>=140 or DP>=90
    y_htn = ((df['SP'] >= 140) | (df['DP'] >= 90)).astype(int).values

    # 4) Features (drop targets)
    X = df.drop(columns=['SP', 'DP'])

    # 5) Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6) Save processed data
    # Features
    X_out = pd.DataFrame(X_scaled, columns=X.columns)
    X_out_path = os.path.join(out_folder, f'X_fold_{fold}_scaled.csv')
    X_out.to_csv(X_out_path, index=False)

    # Labels
    y_df = pd.DataFrame({
        'SP': y_sp,
        'DP': y_dp,
        'HTN': y_htn
    })
    y_out_path = os.path.join(out_folder, f'y_fold_{fold}.csv')
    y_df.to_csv(y_out_path, index=False)

    print(f"Saved features to {X_out_path}")
    print(f"Saved labels   to {y_out_path}")
