import torch
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from torch.cuda.amp import autocast
import torch.multiprocessing as mp

from model.conformer import Conformer
from processing.processor import ConformerProcessor
from processing.lm import KenLanguageModel
from dataset import InferenceDataset, ConformerCollate, ConformerDataset
from checkpoint import load_model
from tqdm import tqdm

import pandas as pd
import fire

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
        rank: Union[str, int],
        world_size: int,
        manifest: str,
        checkpoint: str,
        lm_path: str, 
        batch_size: int = 1,
        fp16: bool = False,
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
        tokenizer_path: str = "./tokenizer/vi.json",
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
        saved_result_path: Optional[str] = None
    ) -> None:
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
        unk_token=unk_token,
        device=rank
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

    load_model(checkpoint, model)
    model.to(rank)
    model.eval()

    lm = KenLanguageModel(
        lm_path=lm_path,
        vocab=processor.vocab
    )

    if ".csv" in manifest:
        df = pd.read_csv(manifest)

        collate_fn = ConformerCollate(processor)
        dataset = ConformerDataset(df, processor, num_examples=None)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=collate_fn)

        predicts = []
        for (inputs, lengths, sorted_indices) in tqdm(dataloader, leave=False):
            with torch.inference_mode():
                with autocast(enabled=fp16):
                    outputs, lengths = model(inputs, lengths)
                    preds = lm.decode_batch(outputs.cpu().numpy(), lengths.cpu().numpy(), decode_func=processor.spec_decode)
                    predicts += [preds[i] for i in sorted_indices]

        df['prediction'] = predicts
        if saved_result_path is not None:
            df.to_csv(saved_result_path, index=False)
    
    else:
        audio = processor.read_audio(manifest).unsqueeze(0)
        audio = processor.mel_spectrogram(audio)

        with torch.inference_mode():
            with autocast(enabled=fp16):
                outputs, _ = model(audio, None)
                pred = lm.decode_batch(outputs.cpu().numpy(), lengths.cpu().numpy(), decode_func=processor.spec_decode)

        print("Model Prediction:")
        print(pred)

def main(
        manifest: str,
        checkpoint: str,
        lm_path: str, 
        batch_size: int = 1,
        fp16: bool = False,
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
        tokenizer_path: str = "./tokenizer/vi.json",
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
        saved_result_path: Optional[str] = None
    ):
    fp16 = bool(fp16 == 1)
    
    n_gpus = 0
    if device == 'cuda':
        n_gpus = torch.cuda.device_count()
    
    if device == 'cpu' or n_gpus == 1:
        if n_gpus == 1:
            device = 0
        infer(
            device, n_gpus, manifest,
            checkpoint, lm_path, batch_size, fp16,
            sampling_rate, n_mels, n_fft, win_length, hop_length, fmin, fmax, mel_norm, mel_scale,
            tokenizer_path, pad_token, delim_token, unk_token,
            n_conformer_blocks, d_model, n_heads, kernel_size, lstm_hidden_dim, n_lstm_layers, saved_result_path
        )
    else:
        mp.spawn(
            fn=infer,
            args=(
                device, n_gpus, manifest,
                checkpoint, lm_path, batch_size, fp16,
                sampling_rate, n_mels, n_fft, win_length, hop_length, fmin, fmax, mel_norm, mel_scale,
                tokenizer_path, pad_token, delim_token, unk_token,
                n_conformer_blocks, d_model, n_heads, kernel_size, lstm_hidden_dim, n_lstm_layers, saved_result_path
            ),
            nprocs=n_gpus,
            join=True
        )

if __name__ == '__main__':
    fire.Fire(main)