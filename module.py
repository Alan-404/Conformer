import torch
from torch.optim import Adam

import lightning as L

import torch.optim.lr_scheduler as lr_scheduler

from preprocessing.processor import ConformerProcessor
from src.conformer import Conformer

from src.loss import ctc_loss

class ConformerModule(L.LightningModule):
    def __init__(self, processor: ConformerProcessor, encoder_n_layers: int, encoder_dim: int, heads: int, kernel_size: int, decoder_n_layers: int, decoder_dim: int, dropout_rate: float, lr: float = 1e-4) -> None:
        super().__init__()

        self.processor = processor

        self.model = Conformer(
            vocab_size=len(processor.dictionary),
            n_mel_channels=processor.num_mels,
            encoder_n_layers=encoder_n_layers,
            encoder_dim=encoder_dim,
            heads=heads,
            kernel_size=kernel_size,
            decoder_n_layers=decoder_n_layers,
            decoder_dim=decoder_dim,
            dropout_rate=dropout_rate
        )

        self.lr = lr

    def training_step(self, batch, _: int):
        inputs = batch[0]
        labels = batch[1]

        input_lengths = batch[2]
        target_lengths = batch[3]

        outputs, input_lengths = self.model(inputs, input_lengths)

        loss = ctc_loss(
            outputs,
            labels,
            input_lengths,
            target_lengths,
            blank_id=self.processor.pad_token,
            zero_infinity=True
        )
        
        self.log('train_loss', value=loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]
