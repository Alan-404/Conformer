import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import lightning as L

from model.conformer import Conformer
from model.byol import BYOL
from model.wav2vec2 import Wav2Vec2

from evaluation import ConformerCriterion, ConformerMetric

from typing import Tuple, Callable, Optional
import statistics

class ConformerModule(L.LightningModule):
    def __init__(self, vocab_size: int, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int, dropout_rate: float, pad_token: int, metric_fx: Callable[[str, bool], torch.Tensor], set_augment: bool = True) -> None:
        super().__init__()
        self.metric_fx = metric_fx

        self.model = Conformer(
            vocab_size=vocab_size,
            n_mel_channels=n_mel_channels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )

        self.train_loss = []
        self.val_loss = []
        self.val_score = []
        
        self.criterion = ConformerCriterion(blank_id=pad_token)
        self.metric = ConformerMetric()

        self.set_augment = set_augment

        self.save_hyperparameters(ignore=["pad_token", "metric_fx", "set_augment"])
    
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

class Wav2Vec2Module(L.LightningModule):
    def __init__(self, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int, proj_dim: int = 256, num_groups: int = 2, num_vars: int = 320, dropout_rate: float = 0.0, num_negatives: int = 100, diversity_weight: float = 0.1) -> None:
        super().__init__()
        self.model = Wav2Vec2(
            n_blocks=n_blocks,
            n_mel_channels=n_mel_channels,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            proj_dim=proj_dim,
            num_groups=num_groups,
            num_vars=num_vars,
            dropout_rate=dropout_rate
        )

        self.num_groups = num_groups
        self.num_vars = num_vars

        self.train_loss = []

        self.num_negatives = num_negatives
        self.diversity_weight = diversity_weight

        self.criterion = ConformerCriterion()
    
    def training_step(self, batch: Tuple[torch.Tensor], _: int):
        inputs = batch[0]

        input_lengths = batch[1]

        context_features, quantized_features, perplexity, mask_indexes = self.model(inputs, input_lengths)

        batch_size, sequence_length, hidden_size = context_features.size()

        neg_indexes = self._sample_negative_indices(batch_size, sequence_length, self.num_negatives, mask_time_indices=mask_indexes, device=context_features.device)

        negative_quantized_features = quantized_features.view(-1, hidden_size)[neg_indexes.long().view(-1)]
        negative_quantized_features = negative_quantized_features.view(batch_size, sequence_length, -1, hidden_size).permute(2, 0, 1, 3)

        logits = self.compute_contrastive_logits(
            quantized_features[None, :],
            negative_quantized_features,
            context_features
        )

        neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float('-inf')

        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))

        target = ((1 - mask_indexes.long()) * -100).transpose(0, 1).flatten()

        contrastive_loss = F.cross_entropy(logits.float(), target, reduction="sum")

        num_codevectors = self.num_vars * self.num_groups
        diversity_loss = ((num_codevectors - perplexity) / num_codevectors) * mask_indexes.sum()

        loss = contrastive_loss + self.diversity_weight * diversity_loss
        self.train_loss.append(loss.item())
        
        return loss
    
    def _sample_negative_indices(self, batch_size: int, sequence_length: int, num_negatives: int, mask_time_indices: Optional[torch.Tensor] = None, device: str = 'cuda'):
        # generate indices of the positive vectors themselves, repeat them `num_negatives` times
        sequence_length_range = torch.arange(sequence_length, dtype=torch.int, device=device)

        # get `num_negatives` random vector indices from the same utterance
        sampled_negative_indices = torch.zeros(size=(batch_size, sequence_length, num_negatives), dtype=torch.int, device=device)

        if mask_time_indices is None:
            mask_time_indices = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            high = mask_time_indices[batch_idx].sum() - 1
            mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

            feature_indices = torch.broadcast_to(torch.arange(high + 1, device=device)[:, None], (high + 1, num_negatives))
            sampled_indices = torch.randint(0, high, size=(high + 1, num_negatives), device=device)
            # avoid sampling the same positive vector, but keep the distribution uniform
            sampled_indices[sampled_indices >= feature_indices] += 1

            # remap to actual indices
            sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

            # correct for batch size
            sampled_negative_indices[batch_idx] += batch_idx * sequence_length

        return sampled_negative_indices
    
    def compute_contrastive_logits(self, target_features: torch.Tensor, negative_features: torch.Tensor, predicted_features: torch.Tensor, temperature: float = 0.2):
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = F.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # apply temperature
        logits = logits / temperature
        return logits
            
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

class BYOLConformerModule(L.LightningModule):
    def __init__(self, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int, dropout_rate: float, alpha: float = 0.95) -> None:
        super().__init__()
        self.model = BYOL(n_mel_channels=n_mel_channels, n_blocks=n_blocks, d_model=d_model, heads=heads, kernel_size=kernel_size, dropout_rate=dropout_rate, alpha=alpha)
        self.criterion = ConformerCriterion()

        self.train_loss = []
        self.val_loss = []
    
    def training_step(self, batch: Tuple[torch.Tensor], _: int):
        self.model.update_moving_average()
        
        online = batch[0]
        target = batch[1]

        lengths = batch[2]

        online, target = self.model(online, target, lengths)

        loss = self.criterion.l2_norm(online, target)

        self.train_loss.append(loss.item())

        return loss
    
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