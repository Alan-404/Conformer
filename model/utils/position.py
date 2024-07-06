import torch
import torch.nn as nn
import math

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.div_term = nn.Parameter(torch.exp(torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / d_model)).unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_positive = torch.zeros((x.size(1), self.d_model), device=self.div_term.device, dtype=self.div_term.dtype)
        pe_negative = torch.zeros((x.size(1), self.d_model), device=self.div_term.device, dtype=self.div_term.dtype)
        position = torch.arange(0, x.size(1), dtype=self.div_term.dtype, device=self.div_term.device).unsqueeze(1)
        angles = torch.matmul(position, self.div_term)

        pe_positive[:, 0::2] = torch.sin(angles)
        pe_positive[:, 1::2] = torch.cos(angles)
        pe_negative[:, 0::2] = torch.sin(-1 * angles)
        pe_negative[:, 1::2] = torch.cos(-1 * angles)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)

        pe = pe.repeat([x.size(0), 1, 1]) # Repeat by Batch Size
        
        return pe