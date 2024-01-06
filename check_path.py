import pandas as pd
from tqdm import tqdm
import os


df = pd.read_csv('./datasets/data.csv', sep="\t")

paths = df['path'].to_list()

valids = []
for path in tqdm(paths):
    valids.append(os.path.exists(path))

df['valid'] = valids

valid_df = df[df['valid'] == True]

valid_df.to_csv('./valid_data.csv', sep="\t", index=False)