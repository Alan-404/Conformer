import os
import torch
from torch.utils.data import DataLoader

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from processing.processor import ConformerProcessor

from module import BYOLConformerModule
from dataset import UnsupervisedConformerDataset
from processing.noise import OnlineAugment, TargetAugment

import fire

from typing import Tuple


def train(
        data_path: str,
        checkpoint: str = None,
        num_epochs: int = 1,
        batch_size: int = 1,
        num_train: int = None,
        saved_folder: str = "./checkpoints",
        sampling_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        n_mel_channels: int = 80,
        n_blocks: int = 17,
        d_model: int = 512,
        heads: int = 8,
        kernel_size: int = 31,
        dropout_rate: float = 0.1,
        alpha: float = 0.95,
        num_workers: int = 1,
        project_name: str = "unsupervised_conformer_byol"
    ):

    if checkpoint is not None:
        module = BYOLConformerModule.load_from_checkpoint(checkpoint_path=checkpoint)
    else:
        module = BYOLConformerModule(
            n_mel_channels=n_mel_channels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            alpha=alpha
        )

    processor = ConformerProcessor(
        sampling_rate=sampling_rate,
        num_mels=n_mel_channels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax
    )

    online_augment = OnlineAugment()
    target_augment = TargetAugment()

    def get_batch(signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mels, lengths = processor(signals)
        
        return online_augment(mels), processor.mel_spectrogram(target_augment(processor(signals, get_signals=True))), lengths

    dataset = UnsupervisedConformerDataset(manifest_path=data_path, processor=processor, num_examples=num_train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=get_batch)

    callbacks = []
    callbacks.append(ModelCheckpoint(saved_folder, filename="byol_{epoch}", save_on_train_epoch_end=True, save_last=True))

    strategy = 'auto'
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)

    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks, precision='16-mixed', strategy=strategy)
    trainer.fit(module, train_dataloaders=dataloader, ckpt_path=checkpoint)

if __name__ == '__main__':
    fire.Fire(train)