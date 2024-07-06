import os 

import torch
from torch.utils.data import DataLoader

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger

from module import UnsupervisedModule
from dataset import UnsupervisedDataset

from processing.processor import ConformerProcessor

import fire

from typing import Optional, Tuple

def pretrain(
        data_path: str,
        saved_checkpoint: str = "./checkpoint/unsupervised_wav2vec2/",
        checkpoint: Optional[str] = None,
        batch_size: int = 1,
        num_epochs: int = 1,
        sampling_rate: int = 16000, 
        fft_size: int = 400, 
        hop_length: int = 160, 
        win_length: int = 400, 
        fmin: float = 0.0, 
        fmax: float = 8000.0,
        num_mels: int = 80,
        n_blocks: int = 17,
        d_model: int = 512,
        heads: int = 8,
        kernel_size: int = 31,
        dropout_rate: float = 0.1,
        proj_dim: int = 256,
        num_groups: int = 2,
        num_vars: int = 320,
        num_negatives: int = 100,
        num_examples: Optional[int] = None,
        project_name: str = "unsupervised_wav2vec2"
    ):
    processor = ConformerProcessor(
        sampling_rate=sampling_rate,
        num_mels=num_mels,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax
    )

    if checkpoint is not None and os.path.exists(checkpoint):
        module = UnsupervisedModule.load_from_checkpoint(checkpoint)
    else:
        module = UnsupervisedModule(
            n_mel_channels=num_mels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            proj_dim=proj_dim,
            num_groups=num_groups,
            num_vars=num_vars,
            num_negatives=num_negatives
        )

    def get_batch(signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mels, lengths = processor(signals)
        return mels, lengths

    dataset = UnsupervisedDataset(manifest_path=data_path, processor=processor, num_examples=num_examples)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=get_batch)

    callbacks = [
        ModelCheckpoint(dirpath=saved_checkpoint, filename="{epoch}", save_on_train_epoch_end=True, save_last=True),
        EarlyStopping(monitor='train_loss', mode='min')
    ]

    strategy = 'auto'
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True)

    logger = WandbLogger(project=project_name)

    trainer = Trainer(max_epochs=num_epochs, callbacks=callbacks, precision='16-mixed', strategy=strategy, logger=logger)
    
    trainer.fit(module, train_dataloaders=dataloader, ckpt_path=checkpoint)

if __name__ == '__main__':
    fire.Fire(pretrain)