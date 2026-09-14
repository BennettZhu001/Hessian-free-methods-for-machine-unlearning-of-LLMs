"""Synthetic sequence data with deletion requests that arrive over time."""

from __future__ import annotations

from dataclasses import dataclass

import torch

TensorPair = tuple[torch.Tensor, torch.Tensor]


def _shift(sequences: torch.Tensor) -> TensorPair:
    return sequences[:, :-1].contiguous(), sequences[:, 1:].contiguous()


@dataclass(frozen=True)
class SequentialCorpus:
    retain_inputs: torch.Tensor
    retain_targets: torch.Tensor
    deletion_requests: tuple[TensorPair, ...]
    full_inputs: torch.Tensor
    full_targets: torch.Tensor
    vocab_size: int

    def retained_after(self, completed_requests: int) -> TensorPair:
        """Return data remaining after the first ``completed_requests``."""
        if not 0 <= completed_requests <= len(self.deletion_requests):
            raise ValueError("completed_requests is out of range")
        inputs = [self.retain_inputs]
        targets = [self.retain_targets]
        for request_inputs, request_targets in self.deletion_requests[
            completed_requests:
        ]:
            inputs.append(request_inputs)
            targets.append(request_targets)
        return torch.cat(inputs), torch.cat(targets)

    def deleted_through(self, completed_requests: int) -> TensorPair:
        if not 1 <= completed_requests <= len(self.deletion_requests):
            raise ValueError("completed_requests must be between 1 and n_requests")
        selected = self.deletion_requests[:completed_requests]
        return (
            torch.cat([pair[0] for pair in selected]),
            torch.cat([pair[1] for pair in selected]),
        )


def make_sequential_corpus(
    *,
    n_retain: int = 2048,
    n_requests: int = 4,
    examples_per_request: int = 32,
    sequence_length: int = 65,
    vocab_size: int = 256,
    seed: int = 7,
) -> SequentialCorpus:
    """Build modular background sequences and distinct repeated canaries."""
    if min(n_retain, n_requests, examples_per_request) < 1:
        raise ValueError("dataset sizes must be positive")
    if sequence_length < 3 or vocab_size < 16:
        raise ValueError("sequence_length >= 3 and vocab_size >= 16 are required")

    generator = torch.Generator().manual_seed(seed)
    content_size = vocab_size - 3
    starts = torch.randint(0, content_size, (n_retain,), generator=generator)
    steps = torch.randint(1, 8, (n_retain,), generator=generator)
    positions = torch.arange(sequence_length - 1)
    normal = torch.empty((n_retain, sequence_length), dtype=torch.long)
    normal[:, 0] = 1
    normal[:, 1:] = (
        starts[:, None] + steps[:, None] * positions[None, :]
    ) % content_size + 3

    requests: list[TensorPair] = []
    request_sequences: list[torch.Tensor] = []
    for request_index in range(n_requests):
        # Each request has a distinct nonlinear canary, repeated sufficiently
        # often to leave a measurable training signal at this small scale.
        canary = torch.tensor(
            [1]
            + [
                3
                + (
                    (request_index + 3) * j * j
                    + (2 * request_index + 5) * j
                    + 11 * request_index
                )
                % content_size
                for j in range(sequence_length - 1)
            ],
            dtype=torch.long,
        ).repeat(examples_per_request, 1)
        request_sequences.append(canary)
        requests.append(_shift(canary))

    full_sequences = torch.cat([normal, *request_sequences])
    permutation = torch.randperm(full_sequences.shape[0], generator=generator)
    full_inputs, full_targets = _shift(full_sequences[permutation])
    retain_inputs, retain_targets = _shift(normal)
    return SequentialCorpus(
        retain_inputs=retain_inputs,
        retain_targets=retain_targets,
        deletion_requests=tuple(requests),
        full_inputs=full_inputs,
        full_targets=full_targets,
        vocab_size=vocab_size,
    )
