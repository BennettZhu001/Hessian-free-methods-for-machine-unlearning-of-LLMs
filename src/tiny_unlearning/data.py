"""Deterministic synthetic next-token data with an explicit deletion set."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CorpusSplit:
    """Full, retained, and forget tensors for causal language modeling."""

    full_inputs: torch.Tensor
    full_targets: torch.Tensor
    retain_inputs: torch.Tensor
    retain_targets: torch.Tensor
    forget_inputs: torch.Tensor
    forget_targets: torch.Tensor
    vocab_size: int


def _shift(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return sequences[:, :-1].contiguous(), sequences[:, 1:].contiguous()


def make_synthetic_corpus(
    *,
    n_retain: int = 80,
    n_forget: int = 8,
    sequence_length: int = 11,
    vocab_size: int = 19,
    seed: int = 7,
) -> CorpusSplit:
    """Create simple modular sequences plus repeated canary sequences.

    Normal sequences follow one of three modular-increment rules. The deletion
    set contains repeated copies of a distinct canary. Repetition makes the
    deletion effect visible in a tiny experiment without external data.
    """
    if n_retain < 1 or n_forget < 1:
        raise ValueError("n_retain and n_forget must both be positive")
    if sequence_length < 3:
        raise ValueError("sequence_length must be at least 3")
    if vocab_size < 8:
        raise ValueError("vocab_size must be at least 8")

    generator = torch.Generator().manual_seed(seed)
    content_size = vocab_size - 3
    starts = torch.randint(0, content_size, (n_retain,), generator=generator)
    steps = torch.randint(1, 4, (n_retain,), generator=generator)

    normal = torch.empty((n_retain, sequence_length), dtype=torch.long)
    normal[:, 0] = 1  # beginning-of-sequence token
    positions = torch.arange(sequence_length - 1)
    normal[:, 1:] = (
        starts[:, None] + steps[:, None] * positions[None, :]
    ) % content_size + 3

    # The canary intentionally violates the normal local transition rules.
    canary_base = torch.tensor(
        [1]
        + [
            3 + ((5 * j * j + 7 * j + 2) % content_size)
            for j in range(sequence_length - 1)
        ],
        dtype=torch.long,
    )
    forget = canary_base.repeat(n_forget, 1)

    full_sequences = torch.cat([normal, forget], dim=0)
    permutation = torch.randperm(full_sequences.shape[0], generator=generator)
    full_sequences = full_sequences[permutation]

    full_inputs, full_targets = _shift(full_sequences)
    retain_inputs, retain_targets = _shift(normal)
    forget_inputs, forget_targets = _shift(forget)
    return CorpusSplit(
        full_inputs=full_inputs,
        full_targets=full_targets,
        retain_inputs=retain_inputs,
        retain_targets=retain_targets,
        forget_inputs=forget_inputs,
        forget_targets=forget_targets,
        vocab_size=vocab_size,
    )
