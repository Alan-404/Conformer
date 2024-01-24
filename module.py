import torch
import torch.nn as nn
from torch.optim import Adam

import lightning as L

import torch.optim.lr_scheduler as lr_scheduler

from torchmetrics.text import WordErrorRate

from preprocessing.processor import ConformerProcessor
from src.conformer import Conformer
from typing import List, Tuple
import statistics

class ConformerModule(L.LightningModule):
    def __init__(self, processor: ConformerProcessor, encoder_n_layers: int, encoder_dim: int, heads: int, kernel_size: int, decoder_n_layers: int, decoder_dim: int, dropout_rate: float, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
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

        self.train_loss = []
        self.val_loss = []
        self.val_score = []

        self.criterion = nn.CTCLoss(blank=processor.pad_token, reduction='mean', zero_infinity=True)
        self.assessor = WordErrorRate()

    def ctc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs.log_softmax(dim=-1).transpose(0,1), targets, input_lengths, target_lengths)
    
    def training_step(self, batch: Tuple[torch.Tensor], _: int):
        inputs = batch[0]
        labels = batch[1]

        input_lengths = batch[2]
        target_lengths = batch[3]

        outputs, input_lengths = self.model(inputs, input_lengths)

        loss = self.ctc_loss(outputs, labels, input_lengths, target_lengths)

        self.train_loss.append(loss.item())

        return loss
    
    def wer_score(self, preds: List[str] | str, labels: List[str] | str):
        return self.assessor(preds, labels)
    
    def validation_step(self, batch: Tuple[torch.Tensor], _: int):
        inputs = batch[0]
        labels = batch[1]

        input_lengths = batch[2]
        target_lengths = batch[3]

        outputs, input_lengths = self.model(inputs, input_lengths)

        loss = self.ctc_loss(outputs, labels, input_lengths, target_lengths)
        
        score = self.wer_score(self.processor.decode_batch(outputs.cpu().numpy()), self.processor.decode_batch(labels.cpu().numpy()))

        self.val_loss.append(loss.item())
        self.val_score.append(score.item())
        
    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]

    def on_train_epoch_end(self):
        print(f"Train Loss: {(statistics.mean(self.train_loss)):.4f}")
        self.train_loss.clear()

    def on_validation_epoch_end(self):
        print(f"Validation Loss: {(statistics.mean(self.val_loss)):.4f}")
        print(f"Validation Score: {(statistics.mean(self.val_score)):.4f}")

        self.val_loss.clear()
        self.val_score.clear()