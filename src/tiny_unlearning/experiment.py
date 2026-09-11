"""End-to-end training, deletion, unlearning, and evaluation experiment."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .data import make_synthetic_corpus
from .model import TinyCausalTransformer
from .second_order import damped_newton_step, flatten, selected_parameters
from .training import (
    language_model_loss,
    negative_log_likelihood,
    predictive_kl,
    train_full_batch,
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    n_retain: int = 80
    n_forget: int = 8
    sequence_length: int = 11
    vocab_size: int = 19
    d_model: int = 16
    d_ff: int = 32
    training_steps: int = 250
    learning_rate: float = 0.03
    l2: float = 1e-4
    damping: float = 2e-2
    cg_max_iter: int = 30
    cg_relative_tolerance: float = 1e-6
    unlearn_parameters: str = "head"


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return flatten(parameter.detach() for parameter in model.parameters())


def _loss_table(
    full_model: torch.nn.Module,
    unlearned_model: torch.nn.Module,
    retrained_model: torch.nn.Module,
    corpus: Any,
) -> dict[str, dict[str, float]]:
    return {
        "full_data_model": {
            "retain_nll": negative_log_likelihood(
                full_model, corpus.retain_inputs, corpus.retain_targets
            ),
            "forget_nll": negative_log_likelihood(
                full_model, corpus.forget_inputs, corpus.forget_targets
            ),
        },
        "newton_unlearned": {
            "retain_nll": negative_log_likelihood(
                unlearned_model, corpus.retain_inputs, corpus.retain_targets
            ),
            "forget_nll": negative_log_likelihood(
                unlearned_model, corpus.forget_inputs, corpus.forget_targets
            ),
        },
        "from_scratch_retraining_reference": {
            "retain_nll": negative_log_likelihood(
                retrained_model, corpus.retain_inputs, corpus.retain_targets
            ),
            "forget_nll": negative_log_likelihood(
                retrained_model, corpus.forget_inputs, corpus.forget_targets
            ),
        },
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if config.unlearn_parameters not in {"head", "all"}:
        raise ValueError("unlearn_parameters must be 'head' or 'all'")

    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    corpus = make_synthetic_corpus(
        n_retain=config.n_retain,
        n_forget=config.n_forget,
        sequence_length=config.sequence_length,
        vocab_size=config.vocab_size,
        seed=config.seed,
    )
    template = TinyCausalTransformer(
        vocab_size=corpus.vocab_size,
        context_length=config.sequence_length - 1,
        d_model=config.d_model,
        d_ff=config.d_ff,
    )
    initial_state = deepcopy(template.state_dict())

    full_model = deepcopy(template)
    full_training = train_full_batch(
        full_model,
        corpus.full_inputs,
        corpus.full_targets,
        steps=config.training_steps,
        learning_rate=config.learning_rate,
        l2=config.l2,
    )

    # From-scratch reference: same architecture, initialization, optimizer,
    # and number of steps, but with the deletion set omitted.
    retrained_model = deepcopy(template)
    retrained_model.load_state_dict(initial_state)
    retained_training = train_full_batch(
        retrained_model,
        corpus.retain_inputs,
        corpus.retain_targets,
        steps=config.training_steps,
        learning_rate=config.learning_rate,
        l2=config.l2,
    )

    unlearned_model = deepcopy(full_model)
    parameters = selected_parameters(unlearned_model, config.unlearn_parameters)

    def retained_objective() -> torch.Tensor:
        return language_model_loss(
            unlearned_model,
            corpus.retain_inputs,
            corpus.retain_targets,
            l2=config.l2,
            regularized_parameters=parameters,
        )

    started = time.perf_counter()
    newton = damped_newton_step(
        retained_objective,
        parameters,
        damping=config.damping,
        cg_max_iter=config.cg_max_iter,
        cg_relative_tolerance=config.cg_relative_tolerance,
    )
    unlearning_seconds = time.perf_counter() - started

    full_vector = _parameter_vector(full_model)
    unlearned_vector = _parameter_vector(unlearned_model)
    retrained_vector = _parameter_vector(retrained_model)
    total_parameters = sum(parameter.numel() for parameter in full_model.parameters())
    updated_parameters = sum(parameter.numel() for parameter in parameters)

    metrics: dict[str, Any] = {
        "config": asdict(config),
        "dataset": {
            "retained_sequences": config.n_retain,
            "deleted_sequences": config.n_forget,
            "deletion_fraction": config.n_forget / (config.n_retain + config.n_forget),
        },
        "model": {
            "total_parameters": total_parameters,
            "updated_parameters": updated_parameters,
            "updated_fraction": updated_parameters / total_parameters,
        },
        "training": {
            "full_data": asdict(full_training),
            "from_scratch_retraining_reference": asdict(retained_training),
            "unlearning_seconds": unlearning_seconds,
        },
        "cg": asdict(newton),
        "losses": _loss_table(full_model, unlearned_model, retrained_model, corpus),
        "comparison_to_retraining": {
            "retain_predictive_kl": predictive_kl(
                retrained_model, unlearned_model, corpus.retain_inputs
            ),
            "forget_predictive_kl": predictive_kl(
                retrained_model, unlearned_model, corpus.forget_inputs
            ),
            "parameter_l2_full_vs_retrained": float(
                torch.linalg.vector_norm(full_vector - retrained_vector)
            ),
            "parameter_l2_unlearned_vs_retrained": float(
                torch.linalg.vector_norm(unlearned_vector - retrained_vector)
            ),
        },
    }

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
