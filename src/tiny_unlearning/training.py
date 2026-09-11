"""Training and evaluation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TrainResult:
    initial_loss: float
    final_loss: float
    seconds: float


def language_model_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    l2: float = 0.0,
    regularized_parameters: list[nn.Parameter] | None = None,
) -> torch.Tensor:
    logits = model(inputs)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    if l2:
        parameters = (
            list(model.parameters())
            if regularized_parameters is None
            else regularized_parameters
        )
        loss = loss + 0.5 * l2 * sum(
            parameter.square().sum() for parameter in parameters
        )
    return loss


def train_full_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    l2: float,
) -> TrainResult:
    """Deterministic full-batch Adam training."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    started = time.perf_counter()
    with torch.no_grad():
        initial = float(language_model_loss(model, inputs, targets))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model, inputs, targets, l2=l2)
        loss.backward()
        optimizer.step()
    seconds = time.perf_counter() - started
    with torch.no_grad():
        final = float(language_model_loss(model, inputs, targets))
    return TrainResult(initial_loss=initial, final_loss=final, seconds=seconds)


@torch.no_grad()
def negative_log_likelihood(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> float:
    model.eval()
    return float(language_model_loss(model, inputs, targets))


@torch.no_grad()
def predictive_kl(
    reference: nn.Module, candidate: nn.Module, inputs: torch.Tensor
) -> float:
    """Average KL(reference || candidate) across sequences and positions."""
    reference.eval()
    candidate.eval()
    reference_log_probs = F.log_softmax(reference(inputs), dim=-1)
    candidate_log_probs = F.log_softmax(candidate(inputs), dim=-1)
    probabilities = reference_log_probs.exp()
    kl = (probabilities * (reference_log_probs - candidate_log_probs)).sum(dim=-1)
    return float(kl.mean())
