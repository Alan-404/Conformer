import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import torchsummary
import torcheval.metrics.functional as F_metric

from preprocessing.processor import ConformerProcessor
from dataset import ConformerDataset
from src.conformer import Conformer

from typing import Tuple

# import wandb

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
parser.add_argument("--n", type=int, default=17)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--kernel_size", type=int, default=31)
parser.add_argument("--hidden_dim", type=int, default=640)
parser.add_argument("--eps", type=float, default=1e-5)
parser.add_argument("--dropout_rate", type=float, default=0.1)

# Training Config
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--saved_checkpoint", type=str, default="./checkpoints")
parser.add_argument("--train_path", type=str, default="./datasets/train.tsv")
parser.add_argument("--device", type=str, default='cuda')
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
parser.add_argument("--lr", type=float, default=7e-5)
parser.add_argument("--weight_decay", type=float, default=0.0)

# Early Stopping Config
parser.add_argument("--early_stopping_patience", type=int, default=4)

# WanDB Config
parser.add_argument("--wandb_project_name", type=str, default="(STT) Conformer")
parser.add_argument("--wandb_username", type=str, default="tri")

########################################
# Parse Config
args = parser.parse_args()

# wandb.init(project=args.wandb_project_name, name=args.wandb_username)

# Device Config
device = 'cpu'
if args.device == 'cuda' or args.device == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

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
    n=args.n,
    d_model=args.d_model,
    heads=args.heads,
    kernel_size=args.kernel_size,
    hidden_dim=args.hidden_dim,
    eps=args.eps,
    dropout_rate=args.dropout_rate
).to(device)

# Optimizer Setup
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)

# Dataset and DataLoader Setup

def get_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    signals, transcripts = zip(*batch)
    mels, mel_lengths = processor(signals, return_attention_mask=True)
    tokens, token_lengths = processor.tokenize(transcripts)

    return mels, tokens, mel_lengths, token_lengths

train_dataset = ConformerDataset(manifest_path=args.train_path, processor=processor, num_examples=args.num_train)

if args.use_validation:
    if args.val_path is not None:
        val_dataset = ConformerDataset(manifest_path=args.val_path, processor=processor, num_examples=args.num_val)
    else:
        train_dataset, val_dataset = random_split(train_dataset, [1 - args.val_size, args.val_size], generator=torch.Generator().manual_seed(41))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=True, collate_fn=get_batch)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=get_batch)

# Evaluation Functions Setup
def loss_func(outputs: torch.Tensor, output_lengths: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
    return F.ctc_loss(
        outputs.log_softmax(dim=-1).transpose(0, 1),
        targets,
        output_lengths,
        target_lengths,
        blank=processor.pad_token,
        zero_infinity=True
    )

def calculate_score(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    hypothesis = processor.decode_batch(outputs)
    reference = processor.decode_batch(labels, group_token=False)

    score = F_metric.word_error_rate(hypothesis, reference)

    return score

# Train and Validate Processing Setup
def train_step(engine: Engine, batch: Tuple[torch.Tensor]) -> float:
    inputs = batch[0].to(device)
    labels = batch[1].to(device)

    input_lengths = batch[2].to(device)
    target_lengths = batch[3].to(device)

    optimizer.zero_grad()
    outputs, input_lengths = model(inputs, input_lengths)

    assert (input_lengths > target_lengths).all()

    loss = loss_func(outputs, input_lengths, labels, target_lengths)

    loss.backward()
    optimizer.step()

    return loss.item()

def val_step(engine: Engine, batch: Tuple[torch.Tensor]) -> Tuple[float, float]:
    inputs = batch[0].to(device)
    labels = batch[1].to(device)

    input_lengths = batch[2].to(device)
    target_lengths = batch[3].to(device)

    mask = batch[4].to(device)

    with torch.no_grad():
        outputs = model(inputs, mask)

    loss = loss_func(outputs, input_lengths, labels, target_lengths).item()

    score = calculate_score(torch.argmax(outputs, dim=-1), labels)

    return loss, score

# Early Stopping Setup
def val_early_stopping_condition(engine: Engine) -> float:
    return 1 - engine.state.metrics['score']

def train_early_stopping_condition(engine: Engine) -> float:
    return -engine.state.metrics['loss']

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
    'lr_scheduler': scheduler
}

checkpoint_manager = Checkpoint(to_save=to_save, 
                                save_handler=DiskSaver(args.saved_checkpoint, create_dir=True, require_empty=False),
                                n_saved=args.early_stopping_patience + 2,
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
    # wandb.log({
    #     "train_loss": engine.state.metrics['loss'], 
    #     'learning_rate': optimizer.param_groups[0]['lr']
    # }, step=engine.state.epoch)
    
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
    # wandb.log({
    #     'val_loss': engine.state.metrics['loss'], 
    #     'val_score': engine.state.metrics['score'],
    #     'early_stopping_patience': early_stopping_handler.counter
    # }, step=trainer.state.epoch)

    val_loss.reset()
    val_score.reset()

if args.use_validation:
    validator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)

# Load Checkpoint
if args.checkpoint is not None:
    assert os.path.exists(args.checkpoint), f"NOT FOUND CHECKPOINT AT {args.checkpoint}"
    Checkpoint.load_objects(to_save, checkpoint=torch.load(args.checkpoint, map_location=device))

if trainer.state.epoch is not None:
    args.num_epochs += trainer.state.epoch

# Start Training
trainer.run(train_dataloader, max_epochs=args.num_epochs)