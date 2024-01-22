import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

# from torch.multiprocessing import start_processes
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import fire

import torchsummary

from preprocessing.processor import ConformerProcessor
from dataset import ConformerDataset
from src.conformer import Conformer
from src.loss import ctc_loss
from src.metric import WER_score

from typing import Tuple

import wandb

wandb.init(project='conformer', name='trind18')

def train(
        # Processor Config
        vocab_path: str = None, 
        train_path: str = None,
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
        # Model Hyper - Params
        encoder_n_layers: int = 17,
        encoder_dim: int = 512,
        heads: int = 8,
        kernel_size: int = 31,
        decoder_n_layers: int = 1,
        decoder_dim: int = 640,
        dropout_rate: float = 0.1,
        # Optimizer
        lr: float = 1e-4,
        # Train config
        checkpoint: str = None,
        num_train: int = None,
        batch_size: int = 1,
        num_epochs: int = 1,
        saved_checkpoint: str = './checkpoints',
        early_stopping_patience: int = 2,
        set_lr: bool = False,
        # Validation Config
        use_validation: bool = False,
        val_size: float = 0.1,
        val_path: str = None,
        val_batch_size: int = 1
    ):
    assert vocab_path is not None and train_path is not None and os.path.exists(vocab_path) and os.path.exists(train_path)

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
    
    model = Conformer(
        vocab_size=len(processor.dictionary.get_itos()),
        n_mel_channels=processor.num_mels,
        encoder_n_layers=encoder_n_layers,
        encoder_dim=encoder_dim,
        heads=heads,
        kernel_size=kernel_size,
        decoder_n_layers=decoder_n_layers,
        decoder_dim=decoder_dim,
        dropout_rate=dropout_rate
    ).cuda()

    # model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6, betas=[0.9, 0.98], eps=1e-9)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
    scaler = GradScaler()
    
    def get_batch(batch, augment: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signals, transcripts = zip(*batch)
        mels, mel_lengths = processor(signals, return_length=True, set_augment=augment)
        tokens, token_lengths = processor.tokenize(transcripts)

        return mels, tokens, mel_lengths, token_lengths

    train_dataset = ConformerDataset(manifest_path=train_path, processor=processor, num_examples=num_train)

    if use_validation:
        if val_path is not None:
            val_dataset = ConformerDataset(manifest_path=val_path, processor=processor)
        else:
            train_dataset, val_dataset = random_split(train_dataset, lengths=[1 - val_size, val_size], generator=torch.Generator().manual_seed(41))
        
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=lambda batch: get_batch(batch, False))

    # train_sampler = DistributedSampler(dataset=train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: get_batch(batch, True))
    
    def train_step(_: Engine, batch: Tuple[torch.Tensor]) -> float:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()

        input_lengths = batch[2].cuda()
        target_lengths = batch[3].cuda()

        optimizer.zero_grad()

        with autocast():
            outputs, input_lengths = model(inputs, input_lengths)
            loss = ctc_loss(
                outputs,
                labels,
                input_lengths,
                target_lengths,
                blank_id=processor.pad_token,
                zero_infinity=True
            )
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)

        scaler.update()
        
        return loss.item()
    
    def val_step(_: Engine, batch: Tuple[torch.Tensor]) -> Tuple[float, float]:
        inputs = batch[0].cuda()
        labels = batch[1].cuda()

        input_lengths = batch[2].cuda()
        target_lengths = batch[3].cuda()

        with torch.no_grad():
            outputs, input_lengths = model(inputs, input_lengths)
            loss = ctc_loss(
                outputs,
                labels,
                input_lengths,
                target_lengths,
                blank_id=processor.pad_token,
                zero_infinity=True
            )

        score = WER_score(
            preds=processor.decode_batch(outputs),
            labels=processor.decode_batch(labels, group_token=False)
        )

        return loss.item(), score
    
    trainer = Engine(train_step)
    train_loss = RunningAverage(output_transform=lambda x: x)
    train_loss.attach(trainer, 'loss')
    ProgressBar().attach(trainer)

    if use_validation:
        def early_stopping_condition(engine: Engine):
            return -engine.state.metrics['score']
        validator = Engine(val_step)
        val_loss = RunningAverage(output_transform=lambda x: x[0])
        val_loss.attach(validator, 'loss')
        val_score = RunningAverage(output_transform=lambda x: x[1])
        val_score.attach(validator, 'score')
        ProgressBar().attach(validator)
        early_stopping_handler = EarlyStopping(patience=early_stopping_patience, score_function=early_stopping_condition, trainer=trainer)

    to_save = {
        'model': model,
        'optimizer': optimizer,
        'lr_scheduler': scheduler,
        'scaler': scaler
    }

    checkpoint_manager = Checkpoint(to_save=to_save, 
                                save_handler=DiskSaver(saved_checkpoint, create_dir=True, require_empty=False),
                                n_saved=early_stopping_patience + 1,
                                global_step_transform=global_step_from_engine(trainer))
    
    @trainer.on(Events.STARTED)
    def _(engine: Engine) -> None:
        if set_lr:
            optimizer.param_groups[0]['lr'] = lr
        print("\nModel Summary")
        torchsummary.summary(model, depth=5)
        
        print("\n================== Training Information ==================")
        print(f"\tNumber of Samples: {len(engine.state.dataloader.dataset)}")
        print(f"\tBatch Size: {engine.state.dataloader.batch_size}")
        print(f"\tNumber of Batches: {len(engine.state.dataloader)}")
        print(f"\tCurrent Learning Rate: {optimizer.param_groups[0]['lr']}")
        print("==========================================================\n")

        if use_validation:
            print("================== Validation Information ==================")
            print(f"\tNumber of Samples: {len(val_dataset)}")
            print(f"\tBatch Size: {val_batch_size}")
            print(f"\tNumber of Batches: {len(val_dataloader)}")
            print("==========================================================\n")

        model.train()

    @trainer.on(Events.EPOCH_STARTED)
    def _(engine: Engine) -> None:
        print(f"========= Epoch {engine.state.epoch} ============")

    @trainer.on(Events.EPOCH_COMPLETED)
    def _(engine: Engine) -> None:
        print(f"Train Loss: {(engine.state.metrics['loss']):.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        wandb.log({
            "train_loss": engine.state.metrics['loss'], 
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=engine.state.epoch)
        
        scheduler.step()
        train_loss.reset()

        if use_validation:
            validator.run(val_dataloader, max_epochs=1)
        
        print(f"========== Done Epoch {engine.state.epoch} =============\n")
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_manager)

    @trainer.on(Events.COMPLETED)
    def _(_: Engine):
        print(f"\nLast Model Checkpoint is saved at {checkpoint_manager.last_checkpoint}\n")

    
    if use_validation:
        @validator.on(Events.STARTED)
        def _(_: Engine):
            val_loss.reset()
            val_score.reset()
            print("Validating...")

        validator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)
        
        @validator.on(Events.COMPLETED)
        def _(engine: Engine):
            print(f"Val Loss: {(engine.state.metrics['loss']):.4f}")
            print(f"Val Score: {(engine.state.metrics['score']):.4f}")

            wandb.log({
                "val_loss": engine.state.metrics['loss'], 
                'val_score': engine.state.metrics['score']
            }, step=trainer.state.epoch)


    if checkpoint is not None:
        assert os.path.exists(checkpoint), f"NOT FOUND CHECKPOINT AT {checkpoint}"
        Checkpoint.load_objects(to_save, checkpoint=torch.load(checkpoint, map_location='cuda'))

    trainer.run(train_dataloader, max_epochs=num_epochs)

if __name__ == '__main__':
    fire.Fire(train)
