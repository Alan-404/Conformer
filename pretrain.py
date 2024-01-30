import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L

import torch.optim.lr_scheduler as lr_scheduler

from torchaudio.transforms import TimeMasking

from model.utils.convolution import ConvolutionSubsampling
from model.utils.block import ConformerBlock
from model.modules.quantization import Quantization
from model.utils.position import RelativePositionalEncoding
from model.utils.masking import generate_mask

from module import ConformerCriterion

from typing import List, Tuple, Union, Callable, Optional
import statistics

class UnsupervisedConformer(nn.Module):
    def __init__(self, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int) -> None:
        super().__init__()
        self.subsampling = ConvolutionSubsampling(channels=d_model)
        self.linear = nn.Linear(in_features=d_model * (((n_mel_channels - 1) // 2 - 1) // 2), out_features=d_model)
        self.rel_pe = RelativePositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([ConformerBlock(d_model=d_model, heads=heads, kernel_size=kernel_size, dropout_rate=0.0) for _ in range(n_blocks)])
        
        self.masking = TimeMasking(time_mask_param=100, p=0.5)
        self.quantizer = Quantization(n_mel_channels=n_mel_channels, d_model=d_model)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        x = self.subsampling(x)

        target = self.quantizer(x)

        context = self.masking(x.transpose(1,2)).transpose(1,2)    
        context = self.linear(context)

        # Mask Generation
        mask = None
        if lengths is not None:
            mask = generate_mask(lengths).to(x.device)
            mask = (mask == 0).unsqueeze(1).unsqueeze(1)
        rel_pos = self.rel_pe(x)
        for layer in self.layers:
            context = layer(context, rel_pos, mask)

        return context, target

class UnsupervisedConformerModule(L.LightningModule):
    def __init__(self, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int) -> None:
        super().__init__()
        self.model = UnsupervisedConformer(
            n_mel_channels=n_mel_channels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads, kernel_size=kernel_size
        )

        self.criterion = ConformerCriterion()

        self.train_loss = []

    
    def training_step(self, batch: Tuple[torch.Tensor], _: int):
        inputs = batch[0]

        input_lengths = batch[1]

        context, target = self.model(inputs, input_lengths)

        loss = self.criterion.contrastive_loss(context, target)

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

def pretrain():
    module = UnsupervisedConformerModule()