"""A minimal one-block, single-head causal transformer."""

from __future__ import annotations

import math

import torch
from torch import nn


class SingleHeadCausalSelfAttention(nn.Module):
    """Explicit attention whose operations support double backward.

    Using explicit score and softmax operations avoids backend-dependent fused
    attention kernels that may omit second derivatives required by HVPs.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        length = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return self.output(torch.matmul(weights, value))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = SingleHeadCausalSelfAttention(d_model, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attended = self.attention(normalized)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x))


class TinyCausalTransformer(nn.Module):
    """Single-head causal LM with one attention block."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int = 16,
        d_ff: int = 32,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.block = TransformerBlock(d_model=d_model, d_ff=d_ff)
        self.output_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.context_length:
            raise ValueError("sequence exceeds configured context length")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.block(x)
        return self.lm_head(self.output_norm(x))
