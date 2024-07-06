import pandas as pd
from processing.processor import ConformerProcessor
from tqdm import tqdm

processor = ConformerProcessor('./vocabulary/vi.json')

df = pd.read_csv('./datasets/train.csv')
df['text'] = df['text'].fillna('')

trans = df['text'].to_list()

graphemes = []
for item in tqdm(trans):
    graphemes.append(" ".join(processor.sentence2graphemes(item)))

df['grapheme'] = graphemes

df.to_csv('./datasets/train.csv', index=False)