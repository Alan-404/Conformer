import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class Quantization(nn.Module):
    def __init__(self, d_model: int, proj_dim: int, num_groups: int = 2, num_vars: int = 320) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_vars = num_vars
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, proj_dim // self.num_groups)
        )
        self.weight_proj = nn.Linear(d_model, self.num_groups * self.num_vars)
        self.temperature = 2

    def _compute_perplexity(self, probs: torch.Tensor):
        marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity
    
    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        codevector_probs = F.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type(hidden_states.dtype)
        
        codevector_soft_dist = F.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )

        perplexity = self._compute_perplexity(codevector_soft_dist)
        
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)

        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors

        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)

        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity