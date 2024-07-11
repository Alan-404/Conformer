import os

import torch

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import torchsummary

from processing.processor import ConformerProcessor
from model.conformer import Conformer
from evaluation import ConformerCriterion, ConformerMetric
from dataset import ConformerDataset, ConformerCollate
from manager import CheckpointManager

from tqdm import tqdm
from typing import Optional
import statistics
import wandb

import fire

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    print(f"Initialize Thread at {rank+1}/{world_size}")

def cleanup():
    dist.destroy_process_group()

def train(
        rank: int,
        world_size: int,
        # Train Config
        train_path: str,
        num_train_samples: Optional[int] = None,
        num_epochs: int = 1,
        train_batch_size: int = 1,
        lr: float = 7e-5,
        fp16: bool = False,
        # Checkpoint Config
        checkpoint: Optional[str] = None,
        saved_folder: str = "./checkpoints",
        n_saved_checkpoints: int = 3,
        saved_checkpoint_after: int = 1,
        # Valdation Config
        val_path: Optional[str] = None,
        val_batch_size: int = 1,
        num_val_samples: Optional[int] = None,
        # Processor Config
        sampling_rate: int = 16000,
        num_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        fmin: float = 0.,
        fmax: float = 8000.,
        tokenizer_path: str = "./tokenizer/vietnamese.json",
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
        dropout_rate: float = 0.1,
        # Logger Config
        logger_project: str = "STT_Conformer",
        logger_name: Optional[str] = None,
    ):
    # assert checkpoint is None or os.path.exists(checkpoint)
    if os.path.exists(saved_folder) == False:
        os.makedirs(saved_folder)
    checkpoint_manager = CheckpointManager(saved_folder, n_saved_checkpoints)

    if world_size > 1:
        setup(rank, world_size)

    if rank == 0:
        wandb.init(project=logger_project, name=logger_name)

    processor = ConformerProcessor(
        sampling_rate=sampling_rate,
        path=tokenizer_path,
        pad_token=pad_token,
        delim_token=delim_token,
        unk_token=unk_token
    )
    
    model = Conformer(
        vocab_size=len(processor.vocab),
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        n_mel_channels=num_mels,
        n_conformer_blocks=n_conformer_blocks,
        d_model=d_model,
        n_heads=n_heads,
        kernel_size=kernel_size,
        lstm_hidden_dim=lstm_hidden_dim,
        n_lstm_layers=n_lstm_layers,
        dropout_rate=dropout_rate
    )

    if rank == 0:
        torchsummary.summary(model)
    
    model.to(rank)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.98], eps=1e-9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    global_steps = 0
    n_epochs = 0
    if checkpoint is not None:
        if os.path.exists(checkpoint):
            global_steps, n_epochs = checkpoint_manager.load_checkpoint(checkpoint, model, optimizer, scheduler)
            print(f"Loaded Model from checkpoint: {checkpoint}")
        else:
            print("Checkpoint is not found, your model will be randomly initilized weights")

    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    collate_fn = ConformerCollate(processor=processor, training=True)

    train_dataset = ConformerDataset(manifest_path=train_path, processor=processor, num_examples=num_train_samples, training=True)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, collate_fn=collate_fn)

    if val_path is not None and os.path.exists(val_path):
        val_dataset = ConformerDataset(manifest_path=val_path, processor=processor, num_examples=num_val_samples)
        val_sampler = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        val_dataloader = DataLoader(train_dataset, batch_size=val_batch_size, sampler=val_sampler, collate_fn=collate_fn, shuffle=(~(world_size > 1)))

    scaler = GradScaler(enabled=fp16)
    criterion = ConformerCriterion(blank_id=processor.pad_id)
    metric = ConformerMetric()

    for epoch in range(num_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)
        
        if rank == 0:
            train_losses = []

            val_losses = []
            val_scores = []

            print(f"Epoch {epoch + 1}")

        model.train()
        for _, (inputs, tokens, input_lengths, token_lengths) in enumerate(tqdm(train_dataloader, leave=False)):
            inputs = inputs.to(rank)
            tokens = tokens.to(rank)
            input_lengths = input_lengths.to(rank)
            token_lengths = token_lengths.to(rank)

            with autocast(enabled=fp16):
                inputs, input_lengths = model(inputs, input_lengths)

                with autocast(enabled=False):
                    loss = criterion.ctc_loss(inputs, tokens, input_lengths, token_lengths)
                    assert torch.isnan(loss) == False

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            scaler.step(optimizer)

            scaler.update()

            if rank == 0:
                train_losses.append(loss.item())

            global_steps += 1
        n_epochs += 1
        scheduler.step()
        
        if val_path is not None:
            model.eval()
            for _, (mels, tokens, mel_lengths, token_lengths) in enumerate(tqdm(val_dataloader, leave=False)):
                mels = mels.to(rank)
                tokens = tokens.to(rank)
                mel_lengths = mel_lengths.to(rank)
                token_lengths = token_lengths.to(rank)

                with torch.no_grad():
                    with autocast(enabled=fp16):
                        outputs, output_lengths = model(mels, mel_lengths)
                
                loss = criterion.ctc_loss(outputs, tokens, output_lengths, token_lengths).item()
                wer_score = metric.wer_score(processor.decode_batch(outputs), processor.decode_batch(tokens, group_token=False))

                val_losses.append(loss)
                val_scores.append(wer_score)

        if rank == 0:
            if n_epochs % saved_checkpoint_after == saved_checkpoint_after - 1 or n_epochs == num_epochs - 1:
                checkpoint_manager.save_checkpoint(model, optimizer, scheduler, global_steps, n_epochs)
            train_loss = statistics.mean(train_losses)

            if val_path is not None:
                val_loss = statistics.mean(val_losses)
                val_score = statistics.mean(val_scores)

            print("--------------------------------")
            print(f"Train Loss: {(train_loss):.4f}")
            wandb.log({
                'train_loss': train_loss
            }, global_steps)
            
            if val_path is not None:
                print(f"Val Loss: {(val_loss):.4f}")
                print(f"Val Score: {(val_score):.4f}")
                wandb.log({
                    'val_loss': val_loss,
                    'val_score': val_score
                }, global_steps)

            print("\n")

    if world_size > 1:
        cleanup()

def main(
        # Train Config
        train_path: str,
        num_train_samples: Optional[int] = None,
        num_epochs: int = 1,
        train_batch_size: int = 1,
        lr: float = 7e-5,
        fp16: bool = False,
        # Checkpoint Config
        checkpoint: Optional[str] = None,
        saved_folder: str = "./checkpoints",
        n_saved_checkpoints: int = 3,
        saved_checkpoint_after: int = 1,
        # Valdation Config
        val_path: Optional[str] = None,
        val_batch_size: int = 1,
        num_val_samples: Optional[int] = None,
        # Processor Config
        sampling_rate: int = 16000,
        num_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        fmin: float = 0.,
        fmax: float = 8000.,
        tokenizer_path: str = "./tokenizer/vietnamese.json",
        pad_token: str = "<PAD>",
        delim_token: str = "|",
        unk_token: str = "<UNK>",
        # Model Config
        n_blocks: int = 17,
        d_model: int = 512,
        n_heads: int = 8,
        kernel_size: int = 31,
        hidden_dim: int = 640,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        # Logger Config
        logger_project: str = "STT_Conformer",
        logger_name: Optional[str] = None,
    ):
    if torch.cuda.is_available() == False:
        raise("CUDA is required")
    n_gpus = torch.cuda.device_count()

    if n_gpus == 1:
        train(
            0, n_gpus,
            train_path, num_train_samples, num_epochs, train_batch_size,
            lr, bool(fp16), checkpoint, saved_folder, n_saved_checkpoints, saved_checkpoint_after,
            val_path, val_batch_size, num_val_samples,
            sampling_rate, num_mels, n_fft, hop_length, win_length, fmin, fmax, tokenizer_path, pad_token, delim_token, unk_token,
            n_blocks, d_model, n_heads, kernel_size, hidden_dim, n_layers, dropout_rate,
            logger_project, logger_name
        )
    else:
        mp.spawn(
            train,
            args=(
                n_gpus,
                train_path, num_train_samples, num_epochs, train_batch_size,
                lr, bool(fp16), checkpoint, saved_folder, n_saved_checkpoints, saved_checkpoint_after,
                val_path, val_batch_size, num_val_samples,
                sampling_rate, num_mels, n_fft, hop_length, win_length, fmin, fmax, tokenizer_path, pad_token, delim_token, unk_token,
                n_blocks, d_model, n_heads, kernel_size, hidden_dim, n_layers, dropout_rate,
                logger_project, logger_name
            ),
            nprocs=n_gpus,
            join=True
        )

if __name__ == '__main__':
    fire.Fire(main)