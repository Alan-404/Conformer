import torch
import torch.nn as nn

from model.utils.convolution import ConvolutionSubsampling
from model.utils.block import ConformerBlock
from model.utils.position import RelativePositionalEncoding
from model.utils.masking import generate_mask

import copy

from typing import Optional

class BYOL(nn.Module):
    def __init__(self, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int, dropout_rate: float, alpha: float=0.95) -> None:
        super().__init__()
        self.online_network = Network(
            n_mel_channels=n_mel_channels,
            n_blocks=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )
        self.target_network = self.copy_network()

        self.predictor = MLP(dim=d_model)

        self.update_handler = EMA(alpha=alpha)

        self.freeze_target()

    def freeze_target(self):
        for params in self.target_network.parameters():
            params.requires_grad = False
    
    @torch.no_grad()
    def copy_network(self):
        return copy.deepcopy(self.online_network)

    @torch.no_grad()
    def update_moving_average(self):
        for online_params, target_params in zip(self.online_network.parameters(), self.target_network.parameters()):
            old_weights, new_weights = target_params.data, online_params.data
            target_params = self.update_handler.update_average(old_weights, new_weights)

    def forward(self, online_item: torch.Tensor, target_item: torch.Tensor, lengths: torch.Tensor):
        online_output = self.online_network(online_item, lengths)
        online_output = self.predictor(online_output)

        with torch.no_grad():
            target_output = self.target_network(target_item, lengths)

        return online_output, target_output

class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_features=dim ,out_features=4*dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(in_features=4*dim, out_features=dim)

    def forward(self, x: torch.Tensor):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.out(x)
        return x

class EMA:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def update_average(self, old: torch.Tensor, new: torch.Tensor):
        if old is None:
            return new
        return self.alpha * old + (1 - self.alpha) * new

class Network(nn.Module):
    def __init__(self, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.subsampling = ConvolutionSubsampling(channels=d_model)
        self.linear = nn.Linear(in_features=d_model * (((n_mel_channels - 1) // 2 - 1) // 2), out_features=d_model)
        self.rel_pe = RelativePositionalEncoding(d_model=d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model=d_model, heads=heads, kernel_size=kernel_size, dropout_rate=dropout_rate) for _ in range(n_blocks)])

        self.projector = MLP(dim=d_model)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, lengths = self.subsampling(x, lengths)

        x = self.linear(x)

        mask = None
        if lengths is not None:
            mask = (generate_mask(lengths).to(x.device) == 0)[:, None, None, :]

        rel_pos = self.rel_pe(x)
        for layer in self.blocks:
            x = layer(x, rel_pos, mask)

        return x