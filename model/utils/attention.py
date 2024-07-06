import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.attention = RelativeMultiHeadAttention(d_model=d_model, heads=heads, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, pos_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.layer_norm(x)
        x = self.attention(x, x, x, pos_embedding, mask)
        x = self.dropout(x)
        return x

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout_rate: float = 0.0):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.head_samples = int(d_model / heads)
        
        self.sqrt_dim = math.sqrt(self.head_samples)

        self.query_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.key_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.value_proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.pos_proj = nn.Linear(in_features=d_model, out_features=d_model)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.content_bias = nn.Parameter(torch.Tensor(self.heads, self.head_samples))
        self.position_bias = nn.Parameter(torch.Tensor(self.heads, self.head_samples))

        self.out_proj = nn.Linear(d_model, d_model)

        self.mask_value = None

        torch.nn.init.xavier_uniform_(self.content_bias)
        torch.nn.init.xavier_uniform_(self.position_bias)

    def scaled_dot_product_relative_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # q x k^T
        content_score = torch.matmul((q + self.content_bias).transpose(1, 2), k.transpose(2, 3))

        # Relative Postional Encoding with score
        pos_score = torch.matmul((q + self.position_bias).transpose(1, 2), pos_embedding)
        pos_score = self._relative_shift(pos_score)

        # Accumulate score and Scale
        attention_score = (content_score + pos_score) / self.sqrt_dim

        # (Optional) Apply Mask
        if mask is not None:
            if self.mask_value is None:
                self.mask_value = torch.iinfo(q.dtype).min
            attention_score = attention_score.masked_fill(mask, self.mask_value)

        # Softmax
        attention_weights = F.softmax(attention_score, dim=-1)

        attention_weights = self.dropout(attention_weights)

        # attention_weights x v
        attention_context = torch.matmul(attention_weights, v)

        return attention_context
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, ctx, _ = q.size()

        # Project and Split Heads
        q = self.query_proj(q).view(batch_size, ctx, self.heads, self.head_samples)
        k = self.key_proj(k).view(batch_size, ctx, self.heads, self.head_samples).permute(0, 2, 1, 3)
        v = self.value_proj(v).view(batch_size, ctx, self.heads, self.head_samples).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.heads, self.head_samples).permute(0, 2, 3, 1)

        # Scaled - dot Product Attention with Relative Position
        attention_context = self.scaled_dot_product_relative_attention(q, k, v, pos_embedding, mask)

        # Concat Heads
        attention_context = attention_context.permute([0, 2, 1, 3])
        attention_context = attention_context.reshape((batch_size, ctx, self.d_model))

        attention_context = self.out_proj(attention_context)

        return attention_context
    
    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score

class MultiHeadLinearAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.dropout_rate = dropout_rate

        assert self.d_model % self.heads == 0

        self.head_samples = self.d_model // self.heads
        self.scaled_dim = math.sqrt(math.sqrt(self.head_samples))

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.linear_out = nn.Linear(in_features=d_model, out_features=d_model)

    def linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = (q / self.scaled_dim).softmax(dim=-2)
        k = (k / self.scaled_dim).softmax(dim=-1).transpose(-1, -2)

        attention_weights = torch.matmul(k, v)
        attention_context = torch.matmul(q, attention_weights)

        return attention_context


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, n_ctx, _ = q.size()

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.reshape((batch_size, n_ctx, self.heads, self.head_samples)).permute([0, 2, 1, 3])
        k = k.reshape((batch_size, n_ctx, self.heads, self.head_samples)).permute([0, 2, 1, 3])
        v = v.reshape((batch_size, n_ctx, self.heads, self.head_samples)).permute([0, 2, 1, 3])

        attention = self.linear_attention(q, k, v)
        attention = attention.permute([0, 2, 1, 3]).reshape((batch_size, n_ctx, self.d_model))
        attention = self.linear_out(attention)

        return attention