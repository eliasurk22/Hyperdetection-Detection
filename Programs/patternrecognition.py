import pandas as pd
import numpy as np
import os
from scipy.stats import skew
from scipy.signal import find_peaks
from numpy.fft import rfft, rfftfreq

base = r'C://Users//Elias John Sabu//Downloads//8th Sem Project//uci2_dataset'  # CHANGE
out_folder = os.path.join(base, 'pattern_feats')
os.makedirs(out_folder, exist_ok=True)

# ---------- Pattern descriptor functions ----------

def lbp_1d(signal):
    s = np.asarray(signal, dtype=float)
    if len(s) < 3:
        return np.zeros(4)
    center = s[1:-1]
    left = s[0:-2]
    right = s[2:]
    codes = ((left > center).astype(int) << 1) + (right > center).astype(int)
    hist, _ = np.histogram(codes, bins=4, range=(0, 4), density=True)
    return hist  # lbp_bin_0..3

def ldp_1d(signal):
    s = np.asarray(signal, dtype=float)
    if len(s) < 3:
        return np.zeros(4)
    g1 = s[1:-1] - s[0:-2]
    g2 = s[2:]   - s[1:-1]
    codes = ((g1 > 0).astype(int) << 1) + (g2 > 0).astype(int)
    hist, _ = np.histogram(codes, bins=4, range=(0, 4), density=True)
    return hist  # ldp_bin_0..3

def ltgp_1d(signal, thresh=0.0):
    s = np.asarray(signal, dtype=float)
    if len(s) < 3:
        return np.zeros(9)
    g = s[1:-1] - s[0:-2]
    codes = np.zeros_like(g, dtype=int)
    codes[g >  thresh] = 2
    codes[g < -thresh] = 0
    codes[(g >= -thresh) & (g <= thresh)] = 1
    hist, _ = np.histogram(codes, bins=9, range=(0, 9), density=True)
    return hist  # ltgp_bin_0..8

def extract_pattern_feats(sig):
    # sig is 1D array of PPG values
    lbp_hist = lbp_1d(sig)
    ldp_hist = ldp_1d(sig)
    ltgp_hist = ltgp_1d(sig)

    feats = {}
    for i, v in enumerate(lbp_hist):
        feats[f'lbp_bin_{i}'] = v
    for i, v in enumerate(ldp_hist):
        feats[f'ldp_bin_{i}'] = v
    for i, v in enumerate(ltgp_hist):
        feats[f'ltgp_bin_{i}'] = v
    return feats

# ---------- Loop over folds and generate pattern features ----------

for fold in [0, 1, 2]:
    sig_fname = f'uci2_signal_fold_{fold}_full.csv'
    sig_path = os.path.join(base, sig_fname)
    if not os.path.exists(sig_path):
        print(f"Missing {sig_path}, skipping")
        continue

    print(f"\nProcessing pattern features for {sig_fname}")
    df_sig = pd.read_csv(sig_path)

    ppg_cols = [c for c in df_sig.columns if c.startswith('ppg_')]
    print("Using", len(ppg_cols), "PPG columns")

    pattern_rows = []
    for _, row in df_sig.iterrows():
        sig = row[ppg_cols].values.astype(float)
        feats = extract_pattern_feats(sig)
        # keep identifiers / labels if you want to merge later
        feats['patient'] = row['patient']
        feats['trial'] = row['trial']
        feats['SP'] = row['SP']
        feats['DP'] = row['DP']
        pattern_rows.append(feats)

    pattern_df = pd.DataFrame(pattern_rows)
    out_path = os.path.join(out_folder, f'pattern_fold_{fold}.csv')
    pattern_df.to_csv(out_path, index=False)
    print("Saved pattern features to", out_path)
