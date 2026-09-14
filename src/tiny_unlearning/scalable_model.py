"""Configurable causal transformer and a convex-in-adapter LoRA head.

The original :mod:`tiny_unlearning.model` remains intentionally minimal.  This
module is the first scale milestone: a multi-layer, multi-head model whose
1M preset is still small enough to audit while exercising realistic training
and distributed code paths.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 256
    context_length: int = 64
    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (
            min(
                self.vocab_size,
                self.context_length,
                self.d_model,
                self.n_heads,
                self.n_layers,
                self.d_ff,
            )
            < 1
        ):
            raise ValueError("all integer model dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class MultiHeadCausalSelfAttention(nn.Module):
    """Explicit causal attention with double-backward-compatible operations."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(x))
        key = self._split_heads(self.key(x))
        value = self._split_heads(self.value(x))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        length = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous()
        attended = attended.view(x.shape[0], length, -1)
        return self.output(attended)


class ScalableTransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadCausalSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class FixedDownLoRAHead(nn.Module):
    """LoRA residual with a fixed down projection and trainable up projection.

    Fixing ``lora_a`` makes the language-model objective convex in ``lora_b``
    when the base transformer is frozen.  That gives CG a positive-semidefinite
    Hessian (made positive-definite by damping), unlike jointly optimizing both
    factors of a bilinear LoRA parameterization.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        seed: int,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        generator = torch.Generator(device=base.weight.device).manual_seed(seed)
        down = torch.empty(
            rank,
            base.in_features,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.kaiming_uniform_(down, a=math.sqrt(5), generator=generator)
        self.lora_a = nn.Parameter(down, requires_grad=False)
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter = torch.nn.functional.linear(x, self.lora_a)
        adapter = torch.nn.functional.linear(adapter, self.lora_b)
        return self.base(x) + self.scale * adapter


class ScalableCausalTransformer(nn.Module):
    """A configurable decoder-only transformer for the 1M milestone."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList(
            ScalableTransformerBlock(config) for _ in range(config.n_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.lm_head: nn.Module = nn.Linear(
            config.d_model, config.vocab_size, bias=True
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("sequence exceeds configured context length")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.output_norm(x))

    def enable_lora_head(
        self, *, rank: int = 8, alpha: float = 16.0, seed: int = 0
    ) -> None:
        """Freeze the base model and install a zero-initialized LoRA head."""
        if isinstance(self.lm_head, FixedDownLoRAHead):
            raise TypeError("LoRA head is already enabled")
        if not isinstance(self.lm_head, nn.Linear):
            raise TypeError("lm_head must be nn.Linear before enabling LoRA")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.lm_head = FixedDownLoRAHead(
            self.lm_head, rank=rank, alpha=alpha, seed=seed
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def model_summary(self) -> dict[str, object]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "config": asdict(self.config),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_fraction": trainable / total,
        }
