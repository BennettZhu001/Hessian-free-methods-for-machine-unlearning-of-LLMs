"""Distributed 1M-parameter sequential-unlearning experiment."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .distributed import (
    DistributedContext,
    average_vector,
    collect_peak_memory,
    shard_tensors,
    train_distributed,
)
from .scalable_model import ScalableCausalTransformer, TransformerConfig
from .second_order import damped_newton_step
from .sequential_data import make_sequential_corpus
from .training import language_model_loss, negative_log_likelihood, predictive_kl


@dataclass(frozen=True)
class ScalableExperimentConfig:
    seed: int = 7
    vocab_size: int = 256
    context_length: int = 64
    d_model: int = 160
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    n_retain: int = 2048
    n_requests: int = 4
    examples_per_request: int = 32
    training_steps: int = 400
    batch_size_per_gpu: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    precision: str = "bfloat16"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    curvature_examples_per_rank: int = 64
    damping: float = 5e-2
    cg_max_iter: int = 20
    cg_relative_tolerance: float = 1e-5
    recovery_steps: int = 20
    recovery_learning_rate: float = 1e-2
    train_retraining_reference: bool = True

    def model_config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )


def load_scalable_config(path: str | Path) -> ScalableExperimentConfig:
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise TypeError("configuration file must contain one JSON object")
    allowed = ScalableExperimentConfig.__dataclass_fields__
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown configuration fields: {', '.join(unknown)}")
    return ScalableExperimentConfig(**values)


def _nll(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> float:
    return negative_log_likelihood(model, inputs.to(device), targets.to(device))


def _kl(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> float:
    return predictive_kl(reference, candidate, inputs.to(device))


def _maximum_seconds(started: float, context: DistributedContext) -> float:
    elapsed = torch.tensor(time.perf_counter() - started, device=context.device)
    if context.distributed:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed)


def _hardware_summary(context: DistributedContext) -> dict[str, Any]:
    if context.device.type == "cuda":
        local = {
            "rank": context.rank,
            "name": torch.cuda.get_device_name(context.device),
            "total_memory_bytes": torch.cuda.get_device_properties(
                context.device
            ).total_memory,
        }
    else:
        local = {"rank": context.rank, "name": "CPU", "total_memory_bytes": 0}
    devices: list[dict[str, Any]] = [local]
    if context.distributed:
        devices = [{} for _ in range(context.world_size)]
        dist.all_gather_object(devices, local)
    return {
        "world_size": context.world_size,
        "distributed_backend": dist.get_backend() if context.distributed else None,
        "devices": devices,
        "pytorch_version": torch.__version__,
    }


def _recovery_attack(
    model: ScalableCausalTransformer,
    deleted_inputs: torch.Tensor,
    deleted_targets: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, float | int]:
    """Try to restore deleted behavior by fine-tuning only the LoRA adapter."""
    attacked = deepcopy(model)
    parameters = attacked.trainable_parameters()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    inputs = deleted_inputs.to(device)
    targets = deleted_targets.to(device)
    before = negative_log_likelihood(attacked, inputs, targets)
    attacked.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(attacked, inputs, targets)
        loss.backward()
        optimizer.step()
    after = negative_log_likelihood(attacked, inputs, targets)
    return {
        "steps": steps,
        "learning_rate": learning_rate,
        "deleted_nll_before_attack": before,
        "deleted_nll_after_attack": after,
        "nll_recovery": before - after,
    }


def run_scalable_experiment(
    config: ScalableExperimentConfig,
    context: DistributedContext,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Run distributed training, sequential LoRA unlearning, and an audit.

    All ranks must call this function. Only rank zero returns and writes metrics.
    """
    torch.manual_seed(config.seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    corpus = make_sequential_corpus(
        n_retain=config.n_retain,
        n_requests=config.n_requests,
        examples_per_request=config.examples_per_request,
        sequence_length=config.context_length + 1,
        vocab_size=config.vocab_size,
        seed=config.seed,
    )
    template = ScalableCausalTransformer(config.model_config())
    initial_state = deepcopy(template.state_dict())

    full_model = deepcopy(template)
    full_training = train_distributed(
        full_model,
        corpus.full_inputs,
        corpus.full_targets,
        context,
        steps=config.training_steps,
        batch_size_per_gpu=config.batch_size_per_gpu,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        precision=config.precision,
        seed=config.seed,
    )

    reference_model: ScalableCausalTransformer | None = None
    reference_training: dict[str, object] | None = None
    if config.train_retraining_reference:
        reference_model = ScalableCausalTransformer(config.model_config())
        reference_model.load_state_dict(initial_state)
        reference_result = train_distributed(
            reference_model,
            corpus.retain_inputs,
            corpus.retain_targets,
            context,
            steps=config.training_steps,
            batch_size_per_gpu=config.batch_size_per_gpu,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            precision=config.precision,
            seed=config.seed,
        )
        reference_training = reference_result.as_dict()

    unlearned_model = deepcopy(full_model)
    unlearned_model.enable_lora_head(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        seed=config.seed,
    )
    unlearned_model.to(context.device)
    unlearned_model.eval()
    trainable = unlearned_model.trainable_parameters()
    sequential_metrics: list[dict[str, Any]] = []

    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    total_unlearning_seconds = 0.0
    for completed in range(1, config.n_requests + 1):
        retained_inputs, retained_targets = corpus.retained_after(completed)
        local_inputs, local_targets = shard_tensors(
            retained_inputs, retained_targets, context
        )
        curvature_size = min(config.curvature_examples_per_rank, local_inputs.shape[0])
        curvature_inputs = local_inputs[:curvature_size]
        curvature_targets = local_targets[:curvature_size]

        def retained_objective(
            inputs: torch.Tensor = curvature_inputs,
            targets: torch.Tensor = curvature_targets,
        ) -> torch.Tensor:
            return language_model_loss(
                unlearned_model,
                inputs,
                targets,
                l2=config.weight_decay,
                regularized_parameters=trainable,
            )

        started = time.perf_counter()
        newton = damped_newton_step(
            retained_objective,
            trainable,
            damping=config.damping,
            cg_max_iter=config.cg_max_iter,
            cg_relative_tolerance=config.cg_relative_tolerance,
            reduce_vector=lambda vector: average_vector(vector, context),
        )
        seconds = _maximum_seconds(started, context)
        total_unlearning_seconds += seconds
        deleted_inputs, deleted_targets = corpus.deleted_through(completed)
        if context.is_primary:
            sequential_metrics.append(
                {
                    "completed_requests": completed,
                    "deleted_sequences": deleted_inputs.shape[0],
                    "retained_sequences": retained_inputs.shape[0],
                    "curvature_sequences_per_rank": curvature_size,
                    "seconds": seconds,
                    "newton_cg": asdict(newton),
                    "retain_nll": _nll(
                        unlearned_model,
                        retained_inputs,
                        retained_targets,
                        context.device,
                    ),
                    "deleted_nll": _nll(
                        unlearned_model,
                        deleted_inputs,
                        deleted_targets,
                        context.device,
                    ),
                }
            )

    hardware = _hardware_summary(context)
    unlearning_allocated, unlearning_reserved = collect_peak_memory(context)
    if not context.is_primary:
        if context.distributed:
            dist.barrier()
        return None

    final_deleted_inputs, final_deleted_targets = corpus.deleted_through(
        config.n_requests
    )
    robustness = _recovery_attack(
        unlearned_model,
        final_deleted_inputs,
        final_deleted_targets,
        steps=config.recovery_steps,
        learning_rate=config.recovery_learning_rate,
        device=context.device,
    )
    comparison: dict[str, float] | None = None
    if reference_model is not None:
        comparison = {
            "retain_predictive_kl_to_retraining": _kl(
                reference_model,
                unlearned_model,
                corpus.retain_inputs,
                context.device,
            ),
            "deleted_predictive_kl_to_retraining": _kl(
                reference_model,
                unlearned_model,
                final_deleted_inputs,
                context.device,
            ),
            "reference_retain_nll": _nll(
                reference_model,
                corpus.retain_inputs,
                corpus.retain_targets,
                context.device,
            ),
            "reference_deleted_nll": _nll(
                reference_model,
                final_deleted_inputs,
                final_deleted_targets,
                context.device,
            ),
        }

    metrics: dict[str, Any] = {
        "experiment": "1m_sequential_lora_unlearning",
        "config": asdict(config),
        "hardware": hardware,
        "model": unlearned_model.model_summary(),
        "dataset": {
            "full_sequences": corpus.full_inputs.shape[0],
            "permanent_retained_sequences": corpus.retain_inputs.shape[0],
            "deletion_requests": config.n_requests,
            "examples_per_request": config.examples_per_request,
        },
        "training": {
            "full_data": full_training.as_dict(),
            "from_scratch_final_retraining_reference": reference_training,
        },
        "sequential_unlearning": {
            "total_seconds": total_unlearning_seconds,
            "peak_allocated_bytes_per_rank": unlearning_allocated,
            "peak_reserved_bytes_per_rank": unlearning_reserved,
            "requests": sequential_metrics,
        },
        "comparison_to_final_retraining": comparison,
        "robustness_audit": robustness,
        "interpretation_boundary": (
            "The recovery fine-tuning attack is a diagnostic, not a certificate of "
            "data removal or robustness. Results must be reported with the exact config."
        ),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metrics, indent=2) + "\n")
    if context.distributed:
        dist.barrier()
    return metrics
