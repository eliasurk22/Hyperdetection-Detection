import os
import pandas as pd

folder = r'C://Users//Elias John Sabu//Downloads//8th Sem Project//uci2_dataset'  # change if needed

for fname in os.listdir(folder):
    if not fname.endswith('.csv'):
        continue
    fpath = os.path.join(folder, fname)
    # Read only the header row (fast)
    df = pd.read_csv(fpath, nrows=0)
    print(f'\nFile: {fname}')
    print(list(df.columns))
