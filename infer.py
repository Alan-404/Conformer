import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from model.conformer import Conformer
from processing.processor import ConformerProcessor
from processing.lm import KenLanguageModel
from dataset import InferenceDataset, ConformerCollate, ConformerDataset
from checkpoint import load_model
from tqdm import tqdm

import pandas as pd

from typing import Optional, Union

def infer_scc(df: pd.DataFrame, checkpoint: str, lm_path: str, type_load: str = 'staff', batch_size: int = 1, device: str = 'cuda', fp16: bool = False):
    dataset = InferenceDataset(df, load_type=type_load, device=device)

    processor = ConformerProcessor(tokenizer_path="./tokenizer/vi.json", device=device)

    model = Conformer(
        vocab_size=len(processor.vocab),
        n_conformer_blocks=17,
        n_mel_channels=80,
        d_model=512,
        n_heads=8,
        kernel_size=31,
        lstm_hidden_dim=640,
        n_lstm_layers=1
    )
    load_model(torch.load(checkpoint, map_location='cpu')['model'], model)
    model.to(device)
    model.eval()

    lm = KenLanguageModel(lm_path=lm_path, vocab=processor.vocab)

    collate_fn = ConformerCollate(processor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    predicts = []
    for (inputs, lengths, sorted_indices) in dataloader:
        with autocast(enabled=fp16):
            with torch.inference_mode():
                outputs, lengths = model(inputs, lengths)
                predicts += lm.decode_batch(outputs.cpu().numpy(), lengths.cpu().numpy(), decode_func=processor.spec_decode)[sorted_indices.cpu().numpy().tolist()]

    df['preds'] = predicts

    return df

def infer(
        manifest: str,
        type: str,
        checkpoint: str,
        lm_path: str, 
        fp16: bool = False,
    ) -> None:
    pass