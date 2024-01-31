import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import lightning as L

from torchmetrics.text import WordErrorRate

from model.conformer import Conformer

from typing import List, Tuple, Union, Callable, Optional
import statistics

class ConformerModule(L.LightningModule):
    def __init__(self, vocab_size: int, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int, n_layers: int, hidden_dim: int, dropout_rate: float, pad_token: int, metric_fx: Callable[[str, bool], torch.Tensor]) -> None:
        super().__init__()
        self.metric_fx = metric_fx

        self.model = Conformer(
            vocab_size=vocab_size,
            n_mel_channels=n_mel_channels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        self.train_loss = []
        self.val_loss = []
        self.val_score = []

        self.criterion = ConformerCriterion(blank_id=pad_token)
        self.metric = ConformerMetric()

        self.save_hyperparameters(ignore=["pad_token", "metric_fx"])
    
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
        score = self.metric.wer_score(self.metric_fx(outputs.cpu().numpy()), self.metric_fx(labels.cpu().numpy(), False))

        self.val_loss.append(loss.item())
        self.val_score.append(score.item())
        
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=3e-5, weight_decay=1e-6, betas=[0.9, 0.98], eps=1e-9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]

    def on_train_epoch_end(self):
        loss = statistics.mean(self.train_loss)
        print(f"Train Loss: {(loss):.4f}")
        print(f"Current Learning Rate: {self.optimizers().param_groups[0]['lr']}")

        self.log("train_loss", loss, rank_zero_only=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], rank_zero_only=True)
        
        self.train_loss.clear()

    def on_validation_epoch_end(self):
        print(f"Validation Loss: {(statistics.mean(self.val_loss)):.4f}")
        print(f"Validation Score: {(statistics.mean(self.val_score)):.4f}")

        self.val_loss.clear()
        self.val_score.clear()

    def freeze_features(self):
        for params in self.model.encoder.subsampling.parameters():
            params.requires_grad = False

class ConformerCriterion:
    def __init__(self, blank_id: Optional[int] = None) -> None:
        self.ctc_criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    def ctc_loss(self, outputs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        return self.ctc_criterion(outputs.log_softmax(dim=-1).transpose(0,1), targets, input_lengths, target_lengths)
    
    def contrastive_loss(self, context: torch.Tensor, target: torch.Tensor):
        k = context.size(0)

        context = context.flatten(start_dim=1)
        target = context.flatten(start_dim=1)

        loss = 0.0

        for i in range(k):
            similaries = []
            for j in range(k):
                similaries.append(F.cosine_similarity(context[i], target[j], dim=-1))
            
            loss += similaries[i] / torch.sum(torch.tensor(similaries))

        return -loss

class ConformerMetric:
    def __init__(self) -> None:
        self.wer_metric = WordErrorRate()

    def wer_score(self, pred: Union[List[str], str], label: Union[List[str], str]) -> torch.Tensor:
        return self.wer_metric(pred, label)
