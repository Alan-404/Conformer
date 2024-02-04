import os
import torch

import torchsummary

import pandas as pd

import fire

from processing.processor import ConformerProcessor
from model.conformer import Conformer

from typing import Tuple
from module import ConformerMetric
from common import map_weights
from tqdm import tqdm

def test(result_folder: str,
         test_path: str,
         vocab_path: str,
         arpa_path: str,
         checkpoint: str,
         num_mels: int = 80,
         sampling_rate: int = 16000,
         fft_size: int = 400,
         hop_length: int = 160,
         win_length: int = 400,
         fmin: float = 0.0,
         fmax: float = 8000.0,
         pad_token: str = "<pad>",
         unk_token: str = "<unk>",
         word_delim_token: str = "|",
         n_blocks: int = 17,
         d_model: int = 512,
         heads: int = 8,
         kernel_size: int = 31,
         n_layers: int = 1,
         hidden_dim: int = 640,
         dropout_rate: float = 0.0,
         num_examples: int = None):
    
    assert os.path.exists(test_path) and os.path.exists(checkpoint)

    if os.path.exists(result_folder) == False:
        os.mkdir(result_folder)

    # Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Processor Setup
    processor = ConformerProcessor(
        vocab_path=vocab_path,
        num_mels=num_mels,
        sampling_rate=sampling_rate,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        pad_token=pad_token,
        unk_token=unk_token,
        word_delim_token=word_delim_token,
        lm_path=arpa_path
    )

    # Model Setup
    model = Conformer(
        vocab_size=len(processor.dictionary),
        n_mel_channels=processor.num_mels,
        n_blocks=n_blocks,
        d_model=d_model,
        heads=heads,
        kernel_size=kernel_size,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate
    ).to(device)

    torchsummary.summary(model)

    checkpoint = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(map_weights(checkpoint['state_dict']))
    else:
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    metric = ConformerMetric()

    df = pd.read_csv(test_path, sep="\t")
    if num_examples is not None:
        df = df[:num_examples]
    df['text'] = df['text'].fillna('')

    answers = df['text'].to_list()
    
    preds = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        start, end = None, None
        if 'start' in df.columns and 'end' in df.columns:
            start = row['start']
            end = row['end']
        
        role = None
        if 'type' in df.columns:
            if row['type'] == 'up':
                role = 0
            elif row['type'] == 'down':
                role = 1

        mel = processor.mel_spectrogram(processor.load_audio(row['path'], start, end, role))

        with torch.no_grad():
            logits = model(mel)
        
        preds.append(processor.decode_beam_search(logits[0].cpu().numpy()))

    print(f"WER Score: {metric.wer_score(preds, answers)}")

    filename = os.path.basename(test_path)
    df.to_csv(f"{result_folder}/{filename}", sep="\t", index=False)
        
if __name__ == '__main__':
    fire.Fire(test)