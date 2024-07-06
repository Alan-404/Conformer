import pandas as pd
from tqdm import tqdm
from typing import Union, List

from processing.processor import ConformerProcessor

import fire

def graphemize(path: Union[str, List[str]], vocab_path: str):
    paths = []
    if type(path) == str:
        paths.append(path)
    elif type(path) == list:
        for item in path:
            paths.append(item)

    processor = ConformerProcessor(vocab_path)

    for item in paths:
        df = pd.read_csv(item)

        text = df['text'].to_list()

        graphemes = []

        for transcript in tqdm(text):
            graphemes.append(processor.sentence2graphemes(transcript))

        df['grapheme'] = graphemes
        df.to_csv(item)

        print("\n")

if __name__ == '__main__':
    fire.Fire(graphemize)