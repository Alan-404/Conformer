import os

import torch
import torch.distributed as distributed
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pandas as pd
from tqdm import tqdm
from typing import Optional

from model.conformer import Conformer
from processing.processor import ConformerProcessor
from dataset import ConformerDataset, ConformerCollate

def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    distributed.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup() -> None:
    distributed.destroy_process_group()

def test(
        rank: int,
        world_size: int,
        # Data
        test_path: str,
        checkpoint: str,
        # Tokenizer:
        tokenizer_path: str,
        pad_token: str = "<PAD>",
        delim_token: str = "|",
        unk_token: str = "<UNK>",
        # Inference Config
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        lm_path: Optional[str] = None,
        # Model Config
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        n_mel_channels: int = 80,
        n_conformer_blocks: int = 17,
        d_model: int = 512,
        n_heads: int = 8,
        kernel_size: int = 31,
        hidden_dim: int = 640,
        n_lstm_layers: int = 1,
        dropout_rate: float = 0.0
    ) -> None:
    assert os.path.exists(test_path) and os.path.exists(checkpoint)
    prompts = pd.read_csv(test_path)

    if world_size > 1:
        setup(rank, world_size)
    
    processor = ConformerProcessor(
        sampling_rate=sample_rate,
        path=tokenizer_path,
        pad_token=pad_token,
        delim_token=delim_token,
        unk_token=unk_token,
        lm_path=lm_path
    )

    model = Conformer(
        vocab_size=len(processor.vocab),
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        n_mel_channels=n_mel_channels,
        n_conformer_blocks=n_conformer_blocks,
        d_model=d_model,
        heads=n_heads,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        n_lstm_layers=n_lstm_layers,
        dropout_rate=dropout_rate
    )

    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])
    model.to(rank)
    model.eval()

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    collate_fn = ConformerCollate()
    
    dataset = ConformerDataset(processor, prompts=prompts, num_examples=num_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)

    for (inputs, input_lengths) in tqdm(dataloader):
        outputs, output_lengths = model.infer(inputs, input_lengths)

        

    if world_size > 1:
        cleanup()