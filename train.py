import os
import torch
from torch.utils.data import DataLoader

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import SingleDeviceStrategy, DDPStrategy
from lightning.pytorch.loggers import WandbLogger

import fire

from module import ConformerModule
from processing.processor import ConformerProcessor
from dataset import ConformerDataset

from torchaudio.transforms import SpecAugment

from typing import Optional, Tuple

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
        n_blocks: int = 17,
        d_model: int = 512,
        heads: int = 8,
        kernel_size: int = 31,
        dropout_rate: float = 0.1,
        # Train config
        num_train: Optional[int] = None,
        batch_size: int = 1,
        num_epochs: int = 1,
        early_stopping_patience: int = 3,
        device: str = "cuda",
        num_workers: int = 1,
        # Augment Config
        set_augment: bool = True,
        n_time_masks: int = 1,
        time_mask_param: int = 10,
        mask_ratio: float = 0.05,
        n_freq_masks: int = 1,
        freq_mask_param: int = 27,
        # Validation Config
        val_path: Optional[str] = None,
        num_val: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        # Tracking Config
        project_name: str = 'speech_to_text_conformer'
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
            vocab_size=len(processor.dictionary),
            n_mel_channels=processor.num_mels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            pad_token=processor.pad_idx,
            metric_fx=processor.decode_batch
        )
    else: 
        module = ConformerModule.load_from_checkpoint(checkpoint, pad_token=processor.pad_token, metric_fx=processor.decode_batch)

    if set_augment:
        spec_augment = SpecAugment(n_time_masks=n_time_masks, time_mask_param=time_mask_param, n_freq_masks=n_freq_masks, freq_mask_param=freq_mask_param, p=mask_ratio)
    
    def get_batch(batch, augment: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        signals, transcripts = zip(*batch)
        mels, mel_lengths = processor(signals)

        if augment:
            mels = spec_augment(mels)

        tokens, token_lengths = processor.tokenize(transcripts)

        return mels, tokens, mel_lengths, token_lengths
    
    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=saved_checkpoint, filename="{epoch}", save_on_train_epoch_end=True, save_last=True))

    if val_path is not None:
        callbacks.append(EarlyStopping(monitor='val_score', verbose=True, mode='min', patience=early_stopping_patience))

    dataset = ConformerDataset(train_path, processor=processor, num_examples=num_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: get_batch(batch, set_augment), num_workers=num_workers)
    
    val_dataloader = None
    if val_path is not None:
        val_dataset = ConformerDataset(val_path, processor=processor, num_examples=num_val)
        if val_batch_size is None:
            val_batch_size = batch_size
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True)

    if device == 'cpu' or not torch.cuda.is_available():
        strategy = SingleDeviceStrategy(device='cpu')
    else:
        if torch.cuda.device_count() == 1:
            strategy = SingleDeviceStrategy(device='cuda')
        else:
            strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)

    logger = WandbLogger(name=project_name, project=project_name)

    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks, precision='16-mixed', strategy=strategy, logger=logger)
    
    trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=val_dataloader, ckpt_path=checkpoint)

if __name__ == '__main__':
    fire.Fire(train)