import pandas as pd
import os

base = r'C:\Users\Elias John Sabu\Downloads\8th Sem Project\uci2_dataset'
X_folder = os.path.join(base, 'preproc_feats')
P_folder = os.path.join(base, 'pattern_feats')

for fold in [0, 1, 2]:
    X = pd.read_csv(os.path.join(X_folder, f'X_fold_{fold}_scaled.csv'))
    y = pd.read_csv(os.path.join(X_folder, f'y_fold_{fold}.csv'))
    P = pd.read_csv(os.path.join(P_folder, f'pattern_fold_{fold}.csv'))

    # Drop identifiers and labels from pattern file, keep only descriptor columns
    P_desc = P.drop(columns=['patient', 'trial', 'SP', 'DP'], errors='ignore')

    # Concatenate features
    X_full = pd.concat([X, P_desc], axis=1)
    X_full.to_csv(os.path.join(X_folder, f'X_fold_{fold}_with_pattern.csv'), index=False)
    print(f"Fold {fold}: X_full shape =", X_full.shape)
