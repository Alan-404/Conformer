import os
import comet_ml
import torch
from torch.utils.data import DataLoader, random_split

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import CometLogger

import fire

import torchsummary

from module import ConformerModule
from preprocessing.processor import ConformerProcessor
from dataset import ConformerDataset

from preprocessing.augment import SpecAugment

from dotenv import load_dotenv
from typing import Optional

load_dotenv()

def train(
        # Processor Config
        train_path: str,
        checkpoint: Optional[str] = None,
        saved_checkpoint: str = './checkpoints/',
        vocab_path: str = './vocabulary/dictionary.json',
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
        n_layers: int = 17,
        d_model: int = 512,
        heads: int = 8,
        kernel_size: int = 31,
        dropout_rate: float = 0.1,
        # Train config
        num_train: Optional[int] = None,
        batch_size: int = 1,
        num_epochs: int = 1,
        early_stopping_patience: int = 3,
        num_workers: int = 1,
        # Augment Config
        set_augment: bool = True,
        freq_augment: int = 27,
        time_augment: int = 10,
        time_mask_ratio: float = 0.05,
        # Validation Config
        use_validation: bool = False,
        val_size: float = 0.1,
        val_path: Optional[str] = None,
        num_val: Optional[int] = None,
        val_batch_size: Optional[int] = None
    ):

    assert os.path.exists(train_path)

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

    if checkpoint is None:
        module = ConformerModule(
            processor=processor,
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )
    else: 
        module = ConformerModule.load_from_checkpoint(checkpoint)

    torchsummary.summary(module.model)
    module.model.train()

    logger = CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name='conformer',
        save_dir=os.environ.get("COMET_WORKDIR"),
        experiment_name='trind18'
    )

    if set_augment:
        spec_augment = SpecAugment(freq_augment=freq_augment, time_augment=time_augment, time_mask_ratio=time_mask_ratio)
    
    def get_batch(batch, augment: bool) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signals, transcripts = zip(*batch)
        mels, mel_lengths = processor(signals, return_length=True)

        if augment:
            mels = spec_augment(mels)

        tokens, token_lengths = processor.tokenize(transcripts)

        return mels, tokens, mel_lengths, token_lengths
    
    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=saved_checkpoint, filename="{epoch}", save_on_train_epoch_end=True, save_top_k=-1))

    if use_validation:
        callbacks.append(EarlyStopping(monitor='val_score', verbose=True, mode='min', patience=early_stopping_patience))

    dataset = ConformerDataset(train_path, processor=processor, num_examples=num_train)
    
    if use_validation:
        if val_batch_size is None:
            val_batch_size = batch_size
        if val_path is not None:
            val_dataset = ConformerDataset(val_path, processor=processor, num_examples=num_val)
        else:
            dataset, val_dataset = random_split(dataset, lengths=[1 - val_size, val_size], generator=torch.Generator().manual_seed(41))
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, collate_fn=lambda batch: get_batch(batch, False), num_workers=num_workers)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: get_batch(batch, set_augment), num_workers=num_workers)

    strategy = 'auto'
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)

    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks, precision='16-mixed', strategy=strategy, logger=logger)
    
    trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=val_dataloader if use_validation else None, ckpt_path=checkpoint)

if __name__ == '__main__':
    fire.Fire(train)