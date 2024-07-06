import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import lightning as L
from lightning.pytorch import strategies

from model.conformer import Conformer
from model.wav2vec2 import Wav2Vec2

import torchsummary

from evaluation import ConformerCriterion, ConformerMetric

import dagshub
import mlflow

from typing import Tuple, Callable, Optional
import statistics

class ConformerModule(L.LightningModule):
    def __init__(self, vocab_size: int, 
                 n_mel_channels: int, 
                 n_blocks: int, 
                 d_model: int, 
                 heads: int, 
                 kernel_size: int, 
                 hidden_dim: int, 
                 n_layers: int, 
                 dropout_rate: float, 
                 pad_token: int, 
                 metric_fx: Callable[[str, bool], torch.Tensor], 
                 project_name: Optional[str] = None, 
                 run_id: Optional[str] = None, 
                 run_name: Optional[str] = None) -> None:
        super().__init__()
        self.model = Conformer(
            vocab_size=vocab_size,
            n_mel_channels=n_mel_channels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )

        self.train_loss = []
        self.val_loss = []
        self.val_score = []

        self.criterion = ConformerCriterion(blank_id=pad_token)
        self.metric = ConformerMetric()

        self.metric_fx = metric_fx

        self.project_name = project_name
        self.run_name = run_name
        self.run_id = run_id

        self.save_hyperparameters(ignore=["pad_token", "metric_fx", "project_name", "run_id", "run_name"])

    def on_train_start(self):
        if self.global_rank == 0:
            dagshub.init(repo_name=os.getenv("DAGSHUB_REPO_NAME"), repo_owner=os.getenv("DAGSHUB_OWNER"), mlflow=True)
            mlflow.set_experiment(experiment_name=self.project_name)
            mlflow.start_run(
                run_id=self.run_id,
                run_name=self.run_name
            )
            
            torchsummary.summary(self.model)
        
        self.model.train()

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

        with torch.no_grad():
            outputs, input_lengths = self.model(inputs, input_lengths)
        
        logits = torch.argmax(outputs, dim=-1).cpu().numpy()

        loss = self.criterion.ctc_loss(outputs, labels, input_lengths, target_lengths)
        score = self.metric.wer_score(self.metric_fx(logits), self.metric_fx(labels.cpu().numpy(), False))

        self.val_loss.append(loss.item())
        self.val_score.append(score.item())
        
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=3e-5, weight_decay=1e-6, betas=[0.9, 0.98], eps=1e-9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]

    def on_train_epoch_end(self):
        if self.trainer.strategy == strategies.DDPStrategy:
            loss = torch.cat(self.all_gather(self.train_loss)).mean().item()
            if len(self.val_loss) != 0:
                val_loss = torch.cat(self.all_gather(self.val_loss)).mean().item()
                val_score = torch.cat(self.all_gather(self.val_score)).mean().item()
        else:
            loss = statistics.mean(self.train_loss)
            if len(self.val_loss) != 0:
                val_loss = statistics.mean(self.val_loss)
                val_score = statistics.mean(self.val_score)


        if self.global_rank == 0:
            print(f"Train Loss: {(loss):.4f}")
            print(f"Current Learning Rate: {self.optimizers().param_groups[0]['lr']}")
            self.log("train_loss", loss, sync_dist=True, rank_zero_only=True)
            self.log('learning_rate', self.optimizers().param_groups[0]['lr'], sync_dist=True, rank_zero_only=True)
            mlflow.log_metric("train_loss", loss, step=self.current_epoch)
            mlflow.log_metric("learning_rate", self.optimizers().param_groups[0]['lr'], step=self.current_epoch)

            if len(self.val_loss) != 0:
                print(f"Validation Loss: {(val_loss):.4f}")
                print(f"Validation Score: {(val_score):.4f}")
                self.log('val_loss', val_loss, sync_dist=True, rank_zero_only=True)
                self.log('val_score', val_score, sync_dist=True, rank_zero_only=True)
                mlflow.log_metric("val_loss", val_loss, step=self.current_epoch)
                mlflow.log_metric("val_score", val_score, step=self.current_epoch)
        
        self.train_loss.clear()
        self.val_loss.clear()
        self.val_score.clear()

    def freeze_features(self):
        for params in self.model.encoder.downsampling_conv.parameters():
            params.requires_grad = False

class UnsupervisedModule(L.LightningModule):
    def __init__(self, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int, proj_dim: int = 256, num_groups: int = 2, num_vars: int = 320, dropout_rate: float = 0.0, num_negatives: int = 100, diversity_weight: float = 0.1, project_name: Optional[str] = None, run_id: Optional[str] = None, run_name: Optional[str] = None) -> None:
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

        self.project_name = project_name
        self.run_id = run_id
        self.run_name = run_name

        self.save_hyperparameters(ignore=['project_name', 'run_id', 'run_name'])

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            dagshub.init(repo_name=os.getenv("DAGSHUB_REPO_NAME"), repo_owner=os.getenv("DAGSHUB_OWNER"), mlflow=True)
            
            mlflow.set_experiment(experiment_name=self.project_name)
            mlflow.start_run(
                run_id=self.run_id,
                run_name=self.run_name
            )
            
            torchsummary.summary(self.model)
        
        self.model.train()
    
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
        optimizer = optim.Adam(params=self.model.parameters(), lr=3e-5, weight_decay=1e-6, betas=[0.9, 0.98], eps=1e-9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]

    def on_train_epoch_end(self):
        if self.trainer.world_size > 1:
            loss = torch.cat(self.all_gather(self.train_loss)).mean().item()
        else:
            loss = statistics.mean(self.train_loss)
        
        if self.global_rank == 0:
            print(f"Train Loss: {(loss):.4f}")
            print(f"Current Learning Rate: {self.optimizers().param_groups[0]['lr']}")

            mlflow.log_metric('train_loss', loss, self.current_epoch)
            mlflow.log_metric('learning_rate', self.optimizers().param_groups[0]['lr'], self.current_epoch)

            self.log("train_loss", loss, rank_zero_only=True, sync_dist=True)
            self.log('learning_rate', self.optimizers().param_groups[0]['lr'], rank_zero_only=True, sync_dist=True)
        
        self.train_loss.clear()
