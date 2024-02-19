import os
import torch
from torch.utils.data import DataLoader

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.loggers.wandb import WandbLogger
from processing.processor import ConformerProcessor

from module import BYOLConformerModule
from dataset import UnsupervisedConformerDataset

from typing import Tuple

def train(
        data_path: str,
        checkpoint: str,
        saved_folder: str,
        num_epochs: int,
        batch_size: int,
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
        alpha: float = 0.99,
        device: str = 'cuda',
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

    def get_batch(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mels = processor(batch, return_length=False)
        return mels

    dataset = UnsupervisedConformerDataset(manifest_path=data_path, processor=processor)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=get_batch)

    callbacks = []
    callbacks.append(ModelCheckpoint(saved_folder, filename="{epoch}", save_on_train_epoch_end=True, save_last=True))

    if device == 'cpu':
        strategy = SingleDeviceStrategy(device=device)
    else:
        if torch.cuda.device_count() == 1:
            strategy = SingleDeviceStrategy(device='cuda')
        else:
            strategy = DDPStrategy(process_group_backend='gloo')

    logger = WandbLogger(
        project=project_name,
        name=os.environ.get("WANDB_USERNAME"),
        save_dir=os.environ.get("WANDB_SAVE_DIR"),
    )

    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks, precision='16-mixed', strategy=strategy, logger=logger)
    trainer.fit(module, train_dataloaders=dataloader, ckpt_path=checkpoint)