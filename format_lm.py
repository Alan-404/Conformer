from preprocessing.processor import ConformerProcessor
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
        word_delim_token: str = "|", 
        num_mels: int = 80, 
        sampling_rate: int = 16000, 
        fft_size: int = 400, 
        hop_length: int = 160, 
        win_length: int = 400, 
        fmin: float = 0.0, 
        fmax: float = 8000.0,
    ):

    assert os.path.exists(txt_path)

    if save_path is None:
        save_path = txt_path

    processor = ConformerProcessor(
        vocab_path=vocab_path,
        unk_token=unk_token,
        pad_token=pad_token,
        word_delim_token=word_delim_token,
        sampling_rate=sampling_rate,
        num_mels=num_mels,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax
    )

    formatted_items = []

    data = io.open(txt_path, encoding='utf-8').read().strip().split("\n")

    print("Formatting...")
    for item in tqdm(data):
        text = item
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