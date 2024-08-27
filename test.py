import os
import torch

from torch.utils.data import DataLoader

import pandas as pd
from model.conformer import Conformer
from dataset import ConformerDataset, ConformerCollate
from processing.processor import ConformerProcessor
from processing.lm import KenLanguageModel
from tqdm import tqdm
from typing import Optional, Union
from evaluation import ConformerMetric
from checkpoint import load_model

import fire

def test(
        test_path: str,
        checkpoint: str,
        lm_path: str, 
        num_samples: Optional[int] = None,
        batch_size: int = 1,
        # Audio Config 
        sampling_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        win_length: Optional[int] = 400,
        hop_length: int = 160,
        fmin: float = 0.0,
        fmax: Optional[float] = 8000.0,
        mel_norm: str = "slaney",
        mel_scale: str = 'slaney',
        # Text Config
        tokenizer_path: str = "./tokenizers/vi.json",
        pad_token: str = "<PAD>",
        delim_token: str = "|",
        unk_token: str = "<UNK>",
        # Model Config
        n_conformer_blocks: int = 17, 
        d_model: int = 512, 
        n_heads: int = 8, 
        kernel_size: int = 31,
        lstm_hidden_dim: int = 640,
        n_lstm_layers: int = 1,
        # Result Config
        saved_result_path: Optional[str] = None,
        # Device
        device: Union[str, int] = 'cuda'
    ):

    assert os.path.exists(test_path) and os.path.exists(checkpoint)

    processor = ConformerProcessor(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=win_length if win_length is not None else n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax if fmax is not None else sampling_rate//2,
        mel_scale=mel_scale,
        norm=mel_norm,
        tokenizer_path=tokenizer_path,
        pad_token=pad_token,
        delim_token=delim_token,
        unk_token=unk_token
    )

    model = Conformer(
        vocab_size=len(processor.vocab),
        n_mel_channels=n_mels,
        n_conformer_blocks=n_conformer_blocks,
        d_model=d_model,
        n_heads=n_heads,
        kernel_size=kernel_size,
        lstm_hidden_dim=lstm_hidden_dim,
        n_lstm_layers=n_lstm_layers,
        dropout_rate=0.0
    )

    load_model(torch.load(checkpoint, map_location='cpu')['model'], model)
    model.to(device)

    lm = KenLanguageModel(
        lm_path=lm_path,
        vocab=processor.vocab
    )

    df = pd.read_csv(test_path)
    if num_samples is not None:
        df = df[:num_samples]

    collate_fn = ConformerCollate(processor)
    dataset = ConformerDataset(df, processor, num_examples=None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    evaluator = ConformerMetric()
    
    predicts = []
    labels = df['text'].to_list()
    for i in range(len(labels)):
        labels[i] = str(labels[i]).upper()
    
    for _, (inputs, lengths) in enumerate(tqdm(dataloader, leave=False)):
        inputs = inputs.to(device)
        lengths = lengths.to(device)

        with torch.inference_mode():
            outputs, lengths = model(inputs, lengths)
            predicts += lm.decode_batch(outputs.cpu().numpy(), lengths.cpu().numpy())
        
    print(f"WER Score: {evaluator.wer_score(predicts, labels)}")
    print(f"CER Score: {evaluator.cer_score(predicts, labels)}")

    df['pred'] = predicts

    if saved_result_path is not None:
        df.to_csv(saved_result_path, index=False)
    
    print("Finish Testing")

if __name__ == '__main__':
    fire.Fire(test)