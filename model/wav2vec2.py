import torch
import torch.nn as nn

from model.modules.encoder import Encoder
from model.modules.quantization import Quantization

from torchaudio.transforms import TimeMasking

from typing import Optional

class Wav2Vec2(nn.Module):
    def __init__(self, n_mel_channels: int, n_blocks: int, d_model: int, heads: int, kernel_size: int, proj_dim: int, dropout_rate: float = 0.0, num_groups: int = 2, num_vars: int = 320) -> None:
        super().__init__()
        self.encoder = Encoder(
            n_mel_channels=n_mel_channels,
            n=n_blocks,
            d_model=d_model,
            heads=heads,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )
        self.projector = nn.Linear(in_features=d_model, out_features=proj_dim)
        self.quantizer = Quantization(d_model=d_model, proj_dim=proj_dim, num_groups=num_groups, num_vars=num_vars)

        self.time_masker = TimeMasking(time_mask_param=10, iid_masks=True, p=0.065)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        x, lengths, hidden_state = self.encoder(x, lengths, return_hidden_state=True)
        x, mask_indexes = self.time_masker(x.transpose(-1, -2)).transpose(-1, -2)
        
        x = self.projector(x)
        
        target, perplexity = self.quantizer(hidden_state, mask_indexes)
        
        return x, target, perplexity
        