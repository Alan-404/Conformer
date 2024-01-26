import torch
import torch.nn as nn
from torch.optim import Adam

import lightning as L

import torch.optim.lr_scheduler as lr_scheduler

from torchmetrics.text import WordErrorRate

from preprocessing.processor import ConformerProcessor
from model.conformer import Conformer

from typing import Any, List, Tuple, Union
import statistics

class ConformerModule(L.LightningModule):
    def __init__(self, processor: ConformerProcessor, n_layers: int, d_model: int, heads: int, kernel_size: int, dropout_rate: float) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=[processor])
        self.processor = processor

        self.model = Conformer(
            vocab_size=len(processor.dictionary),
            n_mel_channels=processor.num_mels,
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )

        self.train_loss = []
        self.val_loss = []
        self.val_score = []

        self.criterion = ConformerCriterion(blank_id=processor.pad_token)
        self.metric = ConformerMetric()
    
    def training_step(self, batch: Tuple[torch.Tensor], _: int):
        inputs = batch[0]
        labels = batch[1]

        input_lengths = batch[2]
        target_lengths = batch[3]

        outputs, input_lengths = self.model(inputs, input_lengths)

        loss = self.criterion.ctc_loss(outputs, labels, input_lengths, target_lengths)

        self.train_loss.append(loss.item())

        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor], _: int):
        inputs = batch[0]
        labels = batch[1]

        input_lengths = batch[2]
        target_lengths = batch[3]

        outputs, input_lengths = self.model(inputs, input_lengths)

        loss = self.criterion.ctc_loss(outputs, labels, input_lengths, target_lengths)
        
        score = self.metric.wer_score(self.processor.decode_batch(outputs.cpu().numpy()), self.processor.decode_batch(labels.cpu().numpy()))

        self.val_loss.append(loss.item())
        self.val_score.append(score.item())
        
    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=3e-5, weight_decay=1e-6, betas=[0.9, 0.98], eps=1e-9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]

    def on_train_epoch_end(self):
        print(f"Train Loss: {(statistics.mean(self.train_loss)):.4f}")
        print(f"Current Learning Rate: {self.optimizers().param_groups[0]['lr']}")
        
        self.train_loss.clear()

    def on_validation_epoch_end(self):
        print(f"Validation Loss: {(statistics.mean(self.val_loss)):.4f}")
        print(f"Validation Score: {(statistics.mean(self.val_score)):.4f}")

        self.val_loss.clear()
        self.val_score.clear()


class ConformerCriterion:
    def __init__(self, blank_id: int) -> None:
        self.criterion = nn.CTCLoss(
            blank=blank_id,
            zero_infinity=True,
            reduction='mean'
        )

    def ctc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> Any:
        return self.criterion(outputs.log_softmax(dim=-1).transpose(0,1), targets, input_lengths, target_lengths)

class ConformerMetric:
    def __init__(self) -> None:
        self.assesor = WordErrorRate()

    def wer_score(self, pred: Union[List[str], str], label: Union[List[str], str]) -> Any:
        return self.assesor(pred, label)
