import torch
import torch.nn as nn
import math

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)).unsqueeze(0)
        self.register_buffer('div_term', torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        angles = torch.matmul(position, self.div_term)

        pe_positive[:, 0::2] = torch.sin(angles)
        pe_positive[:, 1::2] = torch.cos(angles)
        pe_negative[:, 0::2] = torch.sin(-1 * angles)
        pe_negative[:, 1::2] = torch.cos(-1 * angles)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)

        pe = pe.repeat([x.size(0), 1, 1]) # Repeat by Batch Size
        
        return pe.to(x.device)