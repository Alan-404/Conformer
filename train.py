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
import wandb

import fire

def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Initialized Thread {rank+1}/{world_size}")

def cleanup() -> None:
    dist.destroy_process_group()

def validate(
        rank: int,
        world_size: int,
        model: Conformer ,
        dataloader: DataLoader,
        criterion: ConformerCriterion,
        n_steps: int,
        fp16: float,
    ) -> None:

    model.eval()
    for _, (x, y, x_lengths, y_lengths) in enumerate(tqdm(dataloader, leave=False)):
        x = x.to(rank)
        y = y.to(rank)
        x_lengths = x_lengths.to(rank)
        y_lengths = y_lengths.to(rank)

        with torch.no_grad():
            with autocast(enabled=fp16):
                outputs, x_lengths = model(x, x_lengths)
                with autocast(enabled=False):
                    loss = criterion.ctc_loss(outputs, y, x_lengths, y_lengths)
        ctc_loss += loss

    ctc_loss = ctc_loss / len(dataloader)
    if world_size > 1:
        dist.all_reduce(ctc_loss, dist.ReduceOp.AVG)
    
    if rank == 0:
        print("Validation:")
        print(f"Val CTC Loss: {(ctc_loss):.4f}")

        wandb.log({
            'val_ctc_loss': ctc_loss
        }, n_steps)

def train(
        rank: int,
        world_size: int,
        # Train Config
        train_path: str,
        train_batch_size: int = 1,
        num_epochs: int = 1,
        lr: float = 2e-5,
        set_lr: bool = False,
        fp16: float = False,
        num_train_samples: Optional[int] = None,
        # Checkpoint Config
        checkpoint_folder: str = "./checkpoints",
        checkpoint: Optional[str] = None,
        save_checkpoint_after_steps: int = 2000,
        save_checkpoint_after_epochs: int = 1,
        n_saved_checkpoints: int = 3,
        # Validation Config
        val_path: Optional[str] = None,
        val_batch_size: int = 1,
        num_val_samples: Optional[int] = None,
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
        n_conformer_blocks: int = 16, 
        d_model: int = 256, 
        n_heads: int = 4, 
        kernel_size: int = 31,
        lstm_hidden_dim: int = 640,
        n_lstm_layers: int = 1,
        dropout_rate: float = 0.1,
        # Logging Config
        logging: bool = True,
        logging_project: str = "Conformer S2T",
        logging_name: Optional[str] = None
    ) -> None:
    assert os.path.exists(train_path), "Cannot Find the train path"

    if world_size > 1:
        setup(rank, world_size)

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
        dropout_rate=dropout_rate
    )

    if rank == 0:
        if logging:
            wandb.init(project=logging_project, name=logging_name)
        torchsummary.summary(model)
        model.train()
        print("\n")

    model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)

    checkpoint_manager = CheckpointManager(saved_folder=checkpoint_folder, n_savings=n_saved_checkpoints)
    if checkpoint is not None and os.path.exists(checkpoint):
        n_steps, n_epochs = checkpoint_manager.load_checkpoint(checkpoint, model, optimizer, scheduler)
    else:
        n_steps, n_epochs = 0, 0

    if set_lr:
        optimizer.param_groups[0]['lr'] = lr

    collate_fn = ConformerCollate(processor, training=True)

    train_dataset = ConformerDataset(train_path, processor, training=True, num_examples=num_train_samples)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, collate_fn=collate_fn)

    run_validation = False
    if val_path is not None and os.path.exists(val_path):
        if val_batch_size > train_batch_size:
            val_batch_size = train_batch_size
        val_dataset = ConformerDataset(val_path, processor, training=True, num_examples=num_val_samples)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else RandomSampler(val_dataset)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, sampler=val_sampler, collate_fn=collate_fn)
        run_validation = True
    
    criterion = ConformerCriterion(blank_id=processor.pad_id)
    scaler = GradScaler(enabled=fp16)

    for epoch in range(num_epochs):
        if world_size > 1:
            train_dataloader.sampler.set_epoch(epoch)

        ctc_loss = 0.0

        model.train()
        for _, (x, y, x_lengths, y_lengths) in enumerate(tqdm(train_dataloader, leave=False)):
            x = x.to(rank)
            y = y.to(rank)
            x_lengths = x_lengths.to(rank)
            y_lengths = y_lengths.to(rank)

            with autocast(enabled=fp16):
                outputs, x_lengths = model(x, x_lengths)
                with autocast(enabled=False):
                    loss = criterion.ctc_loss(outputs, y, x_lengths, y_lengths)
                    assert torch.isnan(loss) == False

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)

            scaler.update()

            ctc_loss += loss
            n_steps += 1

            if rank == 0 and n_steps % save_checkpoint_after_steps == save_checkpoint_after_steps - 1:
                checkpoint_manager.save_checkpoint(model, optimizer, scheduler, n_steps, n_epochs)
        
        scheduler.step()
        n_epochs += 1

        ctc_loss = ctc_loss / len(train_dataloader)
        if world_size > 1:
            dist.all_reduce(ctc_loss, dist.ReduceOp.AVG)
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}:")
            print("==================================")
            print(f"CTC Loss: {(ctc_loss.item()):.4f}")
            print(f"Current Learning Rate: {current_lr}")

            if logging:
                wandb.log({
                    'ctc_loss': ctc_loss.item(),
                    'learning_rate': current_lr
                }, n_steps)

            if n_epochs % save_checkpoint_after_epochs == save_checkpoint_after_epochs - 1:
                checkpoint_manager.save_checkpoint(model, optimizer, scheduler, n_steps, n_epochs)

        if run_validation:
            validate(
                rank, world_size,
                model, val_dataloader,
                criterion, n_steps, fp16
            )

        if rank == 0:
            print("\n")

    if world_size > 1:
        cleanup()
    print("Finish Training")

def main(
        # Train Config
        train_path: str,
        train_batch_size: int = 1,
        num_epochs: int = 1,
        lr: float = 2e-5,
        set_lr: bool = False,
        fp16: float = False,
        num_train_samples: Optional[int] = None,
        # Checkpoint Config
        checkpoint_folder: str = "./checkpoints",
        checkpoint: Optional[str] = None,
        save_checkpoint_after_steps: int = 2000,
        save_checkpoint_after_epochs: int = 1,
        n_saved_checkpoints: int = 3,
        # Validation Config
        val_path: Optional[str] = None,
        val_batch_size: int = 1,
        num_val_samples: Optional[int] = None,
        # Processor Config
        sampling_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        win_length: Optional[int] = 400,
        hop_length: int = 160,
        fmin: float = 0.0,
        fmax: Optional[float] = 8000.0,
        mel_norm: str = "slaney",
        mel_scale: str = 'slaney',
        # Assessor Config
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
        dropout_rate: float = 0.1,
        # Logging Config
        logging: bool = True,
        logging_project: str = "Conformer S2T",
        logging_name: Optional[str] = None
    ) -> None:

    fp16 = bool(fp16 == 1)
    logging = bool(logging == 1)

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("Not Support CPU training")

    if n_gpus == 1:
        train(
            0, n_gpus,
            train_path, train_batch_size, num_epochs, lr, set_lr, fp16, num_train_samples,
            checkpoint_folder, checkpoint, save_checkpoint_after_steps, save_checkpoint_after_epochs, n_saved_checkpoints,
            val_path, val_batch_size, num_val_samples,
            sampling_rate, n_mels, n_fft, win_length, hop_length, fmin, fmax, mel_norm, mel_scale,
            tokenizer_path, pad_token, delim_token, unk_token,
            n_conformer_blocks, d_model, n_heads, kernel_size, lstm_hidden_dim, n_lstm_layers, dropout_rate,
            logging, logging_project, logging_name
        )
    else:
        mp.spawn(
            fn=train,
            args=(
                n_gpus,
                train_path, train_batch_size, num_epochs, lr, set_lr, fp16, num_train_samples,
                checkpoint_folder, checkpoint, save_checkpoint_after_steps, save_checkpoint_after_epochs, n_saved_checkpoints,
                val_path, val_batch_size, num_val_samples,
                sampling_rate, n_mels, n_fft, win_length, hop_length, fmin, fmax, mel_norm, mel_scale,
                tokenizer_path, pad_token, delim_token, unk_token,
                n_conformer_blocks, d_model, n_heads, kernel_size, lstm_hidden_dim, n_lstm_layers, dropout_rate,
                logging, logging_project, logging_name
            ),
            nprocs=n_gpus,
            join=True
        )

if __name__ == '__main__':
    fire.Fire(main)