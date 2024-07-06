from processing.processor import ConformerProcessor
from typing import Optional
import os
import io
from tqdm import tqdm

import fire

def main(
        vocab_path: str,
        txt_path :str,
        save_path: Optional[str] = None,
        pad_token: str = "<pad>", 
        unk_token: str = "<unk>", 
        word_delim_token: str = "|"
    ):

    assert os.path.exists(txt_path)

    if save_path is None:
        save_path = txt_path

    processor = ConformerProcessor(
        vocab_path=vocab_path,
        unk_token=unk_token,
        pad_token=pad_token,
        word_delim_token=word_delim_token
    )

    formatted_items = []

    data = io.open(txt_path, encoding='utf-8').read().strip().split("\n")

    print("Formatting...")
    for text in tqdm(data):
        text = processor.clean_text(text)
        if hasattr(processor, 'pattern') and len(processor.patterns['replace']) > 0:
            for key in processor.replace_dict:
                text = text.replace(key, processor.replace_dict[key])
        formatted_items.append(text)

    print('Saving...')
    with open(save_path, 'w') as file:
        for item in tqdm(formatted_items):
            file.write(item + "\n")

    print("Done")

if __name__ == '__main__':
    fire.Fire(main)