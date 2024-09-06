import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
import torch.distributed as distributed
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

import pandas as pd
from model.conformer import Conformer
from dataset import ConformerDataset, ConformerCollate
from processing.processor import ConformerProcessor
from processing.lm import KenLanguageModel
from tqdm import tqdm
from typing import Optional, Union, List
from evaluation import ConformerMetric
from checkpoint import load_model

import fire

def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    distributed.init_process_group('nccl', 'env://', world_size=world_size, rank=rank)
    print(f"Initialized Thread {rank+1}/{world_size}")

def cleanup() -> None:
    distributed.destroy_process_group()

def all_gather_list(predictions: List[str], num_items: int, rank: int) -> List[str]:
    gathered_list = [None] * distributed.get_world_size()
    distributed.all_gather_object(gathered_list, predictions)
    gathered_list = [item for sublist in gathered_list for item in sublist]

    if rank == 0:
        results = []
        
        num_steps = num_items // distributed.get_world_size()
        last_idx = num_items - 1
        if num_items % distributed.get_world_size() != 0:
            num_steps += 1

        for index in range(num_steps):
            for rank_idx in range(distributed.get_world_size()):
                idx = index + rank_idx*num_steps
                results.append(gathered_list[idx])
                if idx == last_idx:
                    break
    
        return results

def test(
        device: Union[str, int],
        world_size: int,
        # Data Config
        test_path: str,
        checkpoint: str,
        # CTC Decoder Config
        lm_path: Optional[str] = None, 
        # Inference Config
        num_samples: Optional[int] = None,
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
    if world_size > 1:
        setup(device, world_size)

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
        device=device
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

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device])

    load_model(checkpoint, model, world_size=world_size)
    model.eval()

    ctc_decoder = KenLanguageModel(
        lm_path=lm_path,
        vocab=processor.vocab
    )

    df = pd.read_csv(test_path)
    if num_samples is not None:
        df = df[:num_samples]

    dataset = ConformerDataset(df, processor, num_examples=None)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=device, shuffle=False) if world_size > 1 else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=ConformerCollate(processor, collate_type='test'))

    evaluator = ConformerMetric()
    
    predictions = []
    for (inputs, lengths, sorted_indices) in tqdm(dataloader, leave=False):
        with torch.inference_mode():
            with autocast(enabled=fp16):
                outputs, lengths = model(inputs, lengths)
                preds = ctc_decoder(outputs.cpu().numpy(), lengths.cpu().numpy(), processor.spec_decode)
                predictions += [preds[index] for index in sorted_indices]

    if world_size > 1:
        predictions = all_gather_list(predictions, len(df), device)

    if device == 0 or device == 'cpu':
        labels = df['text'].to_list()
        for i in range(len(labels)):
            labels[i] = str(labels[i]).upper()

        wer_score = evaluator.wer_score(predictions, labels) * 100
        cer_score = evaluator.cer_score(predictions, labels) * 100

        print("Test Results: ")
        print(f"WER Score: {(wer_score):.4f}%")
        print(f"CER Score: {(cer_score):.4f}%")

        if saved_result_path is not None:
            df['prediction'] = predictions
            df.to_csv(saved_result_path, index=False)
        
        print("Finish Testing")

    if world_size > 1:
        cleanup()

def main(
        # Data Config
        test_path: str,
        checkpoint: str,
        # CTC Decoder Config
        lm_path: str, 
        num_samples: Optional[int] = None,
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
        saved_result_path: Optional[str] = None,
        # Device
        device: str = 'cuda'
    ):
    fp16 = bool(fp16 == 1)
    
    n_gpus = 0
    if device == 'cuda':
        n_gpus = torch.cuda.device_count()
    
    if device == 'cpu' or n_gpus == 1:
        if n_gpus == 1:
            device = 0
        test(
            device, n_gpus,
            test_path, checkpoint, 
            lm_path,
            num_samples, batch_size, fp16,
            sampling_rate, n_mels, n_fft, win_length, hop_length, fmin, fmax, mel_norm, mel_scale,
            tokenizer_path, pad_token, delim_token, unk_token,
            n_conformer_blocks, d_model, n_heads, kernel_size, lstm_hidden_dim, n_lstm_layers, saved_result_path
        )
    else:
        mp.spawn(
            fn=test,
            args=(
                n_gpus,
                test_path, checkpoint, 
                lm_path,
                num_samples, batch_size, fp16,
                sampling_rate, n_mels, n_fft, win_length, hop_length, fmin, fmax, mel_norm, mel_scale,
                tokenizer_path, pad_token, delim_token, unk_token,
                n_conformer_blocks, d_model, n_heads, kernel_size, lstm_hidden_dim, n_lstm_layers, saved_result_path
            ),
            nprocs=n_gpus,
            join=True
        )

if __name__ == '__main__':
    fire.Fire(main)