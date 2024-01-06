import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import torchsummary

from preprocessing.grapheme_processor import ConformerProcessor
from dataset import ConformerDataset
from src.conformer import Conformer
from src.loss import ctc_loss
from src.metric import WER_score

from typing import Tuple
import wandb

########################################
from argparse import ArgumentParser

parser = ArgumentParser()

# Audio Processor Config
parser.add_argument("--num_mels", type=int, default=80)
parser.add_argument("--sampling_rate", type=int, default=16000)
parser.add_argument("--fft_size", type=int, default=400)
parser.add_argument("--hop_length", type=int, default=160)
parser.add_argument("--win_length", type=int, default=400)
parser.add_argument("--fmin", type=float, default=0.0)
parser.add_argument("--fmax", type=float, default=8000.0)

# Text Processor Config
parser.add_argument("--vocab_path", type=str, default="./vocab.json")
parser.add_argument("--pad_token", type=str, default="<pad>")
parser.add_argument("--unk_token", type=str, default="<unk>")
parser.add_argument("--word_delim_token", type=str, default="|")

# Model Config
parser.add_argument("--encoder_n_layers", type=int, default=17)
parser.add_argument("--encoder_dim", type=int, default=512)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--kernel_size", type=int, default=31)
parser.add_argument("--eps", type=float, default=1e-5)
parser.add_argument("--decoder_n_layers", type=int, default=1)
parser.add_argument("--decoder_dim", type=int, default=640)
parser.add_argument("--dropout_rate", type=float, default=0.1)

# Training Config
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--saved_checkpoint", type=str, default="./checkpoints")
parser.add_argument("--train_path", type=str, default="./datasets/train.tsv")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_train", type=int, default=None)

# Validation Config
parser.add_argument("--use_validation", type=bool, default=False)
parser.add_argument("--val_path", type=str, default=None)
parser.add_argument("--val_size", type=float, default=0.1)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--num_val", type=int, default=None)

# Optimizer Config
parser.add_argument("--set_lr", type=bool, default=False)
parser.add_argument("--lr", type=float, default=3e-4)

# Early Stopping Config
parser.add_argument("--early_stopping_patience", type=int, default=4)

# # WanDB Config
parser.add_argument("--wandb_project_name", type=str, default="(STT) Conformer")
parser.add_argument("--wandb_username", type=str, default="tri")

########################################
# Parse Config
args = parser.parse_args()

wandb.init(project=args.wandb_project_name, name=args.wandb_username)

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

scaler = GradScaler()

# Processor Setup
processor = ConformerProcessor(
    vocab_path=args.vocab_path,
    num_mels=args.num_mels,
    sampling_rate=args.sampling_rate,
    n_fft=args.fft_size,
    hop_length=args.hop_length,
    win_length=args.win_length,
    fmin=args.fmin,
    fmax=args.fmax,
    pad_token=args.pad_token,
    unk_token=args.unk_token,
    word_delim_token=args.word_delim_token
)

# Model Setup
model = Conformer(
    vocab_size=len(processor.dictionary.get_itos()),
    n_mel_channels=processor.num_mels,
    encoder_n_layers=args.encoder_n_layers,
    encoder_dim=args.encoder_dim,
    heads=args.heads,
    kernel_size=args.kernel_size,
    eps=args.eps,
    decoder_n_layers=args.decoder_n_layers,
    decoder_dim=args.decoder_dim,
    dropout_rate=args.dropout_rate
).to(device)

# Optimizer Setup
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6, betas=[0.9, 0.98], eps=1e-9)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)

# Dataset and DataLoader Setup

def get_batch(batch, augment: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    signals, transcripts = zip(*batch)
    mels, mel_lengths = processor(signals, return_length=True, set_augment=augment)
    tokens, token_lengths = processor.tokenize(transcripts)

    return mels, tokens, mel_lengths, token_lengths

train_dataset = ConformerDataset(manifest_path=args.train_path, processor=processor, num_examples=args.num_train)

if args.use_validation:
    if args.val_path is not None:
        val_dataset = ConformerDataset(manifest_path=args.val_path, processor=processor, num_examples=args.num_val)
    else:
        train_dataset, val_dataset = random_split(train_dataset, [1 - args.val_size, args.val_size], generator=torch.Generator().manual_seed(41))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=True, collate_fn=lambda batch: get_batch(batch, False))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: get_batch(batch, True))

# Train and Validate Processing Setup
def train_step(engine: Engine, batch: Tuple[torch.Tensor]) -> float:
    inputs = batch[0].to(device)
    labels = batch[1].to(device)

    input_lengths = batch[2].to(device)
    target_lengths = batch[3].to(device)

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

def val_step(engine: Engine, batch: Tuple[torch.Tensor]) -> Tuple[float, float]:
    inputs = batch[0].to(device)
    labels = batch[1].to(device)

    input_lengths = batch[2].to(device)
    target_lengths = batch[3].to(device)

    with torch.no_grad():
        outputs = model(inputs, input_lengths)

    loss = ctc_loss(
        outputs,
        labels,
        input_lengths,
        target_lengths,
        blank_id=processor.pad_token,
        zero_infinity=True
    )

    hypothesis = processor.decode_batch(outputs)
    reference = processor.decode_batch(labels, group_token=False)

    score = WER_score(hypothesis, reference)

    return loss, score

# Early Stopping Setup
def val_early_stopping_condition(engine: Engine) -> float:
    return 1 - engine.state.metrics['score']

# Setup Trainer
trainer = Engine(train_step)
train_loss = RunningAverage(output_transform=lambda x: x)
train_loss.attach(trainer, 'loss')
ProgressBar().attach(trainer)

# Setup Validator
validator = Engine(val_step)
ProgressBar().attach(validator)
val_loss = RunningAverage(output_transform=lambda x: x[0])
val_loss.attach(validator, 'loss')
val_score = RunningAverage(output_transform=lambda x: x[1])
val_score.attach(validator, 'score')

if args.use_validation:
    early_stopping_handler = EarlyStopping(patience=args.early_stopping_patience, score_function=val_early_stopping_condition, trainer=trainer)

# Checkpoint Manager Setup
to_save = {
    'model': model,
    'optimizer': optimizer,
    'lr_scheduler': scheduler,
    'scaler': scaler
}

checkpoint_manager = Checkpoint(to_save=to_save, 
                                save_handler=DiskSaver(args.saved_checkpoint, create_dir=True, require_empty=False),
                                n_saved=args.early_stopping_patience + 1,
                                global_step_transform=global_step_from_engine(trainer))

# Trainer Events
@trainer.on(Events.STARTED)
def start_training(engine: Engine) -> None:
    if args.set_lr:
        optimizer.param_groups[0]['lr'] = args.lr
    print("\nModel Summary")
    torchsummary.summary(model, depth=5)
    
    print("\n================== Training Information ==================")
    print(f"\tNumber of Samples: {len(engine.state.dataloader.dataset)}")
    print(f"\tBatch Size: {engine.state.dataloader.batch_size}")
    print(f"\tNumber of Batches: {len(engine.state.dataloader)}")
    print(f"\tCurrent Learning Rate: {optimizer.param_groups[0]['lr']}")
    print("==========================================================\n")

    if args.use_validation:
        print("================== Validation Information ==================")
        print(f"\tNumber of Samples: {len(val_dataset)}")
        print(f"\tBatch Size: {args.val_batch_size}")
        print(f"\tNumber of Batches: {len(val_dataloader)}")
        print("==========================================================\n")

    model.train()

@trainer.on(Events.EPOCH_STARTED)
def start_epoch(engine: Engine) -> None:
    print(f"========= Epoch {engine.state.epoch} ============")

@trainer.on(Events.EPOCH_COMPLETED)
def finish_epoch(engine: Engine) -> None:
    print(f"Train Loss: {(engine.state.metrics['loss']):.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    wandb.log({
        "train_loss": engine.state.metrics['loss'], 
        'learning_rate': optimizer.param_groups[0]['lr']
    }, step=engine.state.epoch)
    
    scheduler.step()
    train_loss.reset()
    if args.use_validation == True:
        validator.run(val_dataloader, max_epochs=1)
    # else:
    #     wandb.log({
    #         'early_stopping_patience': early_stopping_handler.counter
    #     }, step=engine.state.epoch)
    
    print(f"========== Done Epoch {engine.state.epoch} =============\n")

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_manager)

@trainer.on(Events.COMPLETED)
def finish_training(engine: Engine):
    print(f"\nLast Model Checkpoint is saved at {checkpoint_manager.last_checkpoint}\n")

# Validator Events
@validator.on(Events.EPOCH_COMPLETED)
def finish_validating(engine: Engine) -> None:
    print(f"Validation Loss {(engine.state.metrics['loss']):.4f}")
    print(f"Validation WER Score {(engine.state.metrics['score']):.4f}")
    wandb.log({
        'val_loss': engine.state.metrics['loss'], 
        'val_score': engine.state.metrics['score'],
        'early_stopping_patience': early_stopping_handler.counter
    }, step=trainer.state.epoch)

    val_loss.reset()
    val_score.reset()

if args.use_validation:
    validator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)

# Load Checkpoint
if args.checkpoint is not None:
    assert os.path.exists(args.checkpoint), f"NOT FOUND CHECKPOINT AT {args.checkpoint}"
    Checkpoint.load_objects(to_save, checkpoint=torch.load(args.checkpoint, map_location=device))

# Start Training
trainer.run(train_dataloader, max_epochs=args.num_epochs)