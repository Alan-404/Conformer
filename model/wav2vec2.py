import torch
import torch.nn as nn

import torchaudio.transforms as T

from model.utils.convolution import ConvolutionSubsampling
from model.utils.block import ConformerBlock
from model.utils.position import RelativePositionalEncoding
from model.utils.masking import generate_mask

from model.modules.quantization import Quantization

from typing import Optional

class Wav2Vec2(nn.Module):
    def __init__(self, n_blocks: int, n_mel_channels: int, d_model: int, heads: int, kernel_size: int, proj_dim: int, num_groups: int = 2, num_vars: int = 320, time_augment: int = 30, time_mask_ratio: float = 0.065, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.downsampling_conv = ConvolutionSubsampling(channels=d_model)
        self.linear = nn.Linear(in_features=d_model * (((n_mel_channels - 1) // 2 - 1) // 2), out_features=d_model)
        self.rel_pe = RelativePositionalEncoding(d_model=d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model=d_model, heads=heads, kernel_size=kernel_size, dropout_rate=dropout_rate) for _ in range(n_blocks)])

        self.masker = T.TimeMasking(time_mask_param=time_augment, p=time_mask_ratio)

        self.quantization = Quantization(d_model=d_model, n_mel_channels=n_mel_channels, num_codevector_groups=num_groups, num_codevectors_per_group=num_vars, codevector_dim=proj_dim)

        self.hidden_projector = nn.Linear(in_features=d_model, out_features=proj_dim)
        self.quantization_projector = nn.Linear(in_features=proj_dim, out_features=proj_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        x, lengths = self.downsampling_conv(x, lengths)

        context = x.transpose(-1, -2)
        context = self.masker(context)

        mask_indexes = (context.mean(dim=1) != 0)

        target, perplexity = self.quantization(x, mask_indexes)
        target = self.quantization_projector(target)

        context = context.transpose(-1, -2)
        context = self.linear(context)

        mask = None
        if lengths is not None:
            mask = (generate_mask(lengths).to(context.device) == 0)[:, None, None, :]

        rel_pos = self.rel_pe(context)
        for layer in self.blocks:
            context = layer(context, rel_pos, mask)

        context = self.hidden_projector(context)

        return context, target, perplexity, mask_indexes