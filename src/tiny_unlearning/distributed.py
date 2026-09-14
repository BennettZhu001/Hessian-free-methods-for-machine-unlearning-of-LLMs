"""Small, explicit distributed-training utilities for local GPU experiments."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from .training import language_model_loss


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class DistributedTrainResult:
    initial_loss: float
    final_loss: float
    seconds: float
    tokens_per_second: float
    peak_allocated_bytes_per_rank: tuple[int, ...]
    peak_reserved_bytes_per_rank: tuple[int, ...]
    world_size: int
    precision: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def initialize_distributed(
    *, require_world_size: int | None = None
) -> DistributedContext:
    """Initialize from ``torchrun`` environment variables, or run singly."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if require_world_size is not None and world_size != require_world_size:
        raise RuntimeError(
            f"expected WORLD_SIZE={require_world_size}, got {world_size}; "
            "launch with torchrun"
        )

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices exist"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        if require_world_size is not None:
            raise RuntimeError("the requested distributed GPU run requires CUDA")
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(rank, local_rank, world_size, device, distributed)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def average_vector(vector: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Average a detached vector across ranks for distributed HVP/CG."""
    if not context.distributed:
        return vector
    averaged = vector.clone()
    dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
    averaged.div_(context.world_size)
    return averaged


def shard_tensors(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    context: DistributedContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create equal-size deterministic shards, padding only when necessary."""
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must contain the same number of sequences")
    per_rank = math.ceil(inputs.shape[0] / context.world_size)
    padded_size = per_rank * context.world_size
    indices = torch.arange(inputs.shape[0])
    if padded_size > inputs.shape[0]:
        indices = torch.cat([indices, indices[: padded_size - inputs.shape[0]]])
    local_indices = indices[context.rank : padded_size : context.world_size]
    return (
        inputs[local_indices].to(context.device),
        targets[local_indices].to(context.device),
    )


def _global_mean(value: torch.Tensor, context: DistributedContext) -> float:
    result = value.detach().to(context.device)
    if context.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result.div_(context.world_size)
    return float(result)


def collect_peak_memory(
    context: DistributedContext,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if context.device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(context.device)
        reserved = torch.cuda.max_memory_reserved(context.device)
    else:
        allocated = reserved = 0
    if not context.distributed:
        return (allocated,), (reserved,)
    allocated_values: list[int] = [0 for _ in range(context.world_size)]
    reserved_values: list[int] = [0 for _ in range(context.world_size)]
    dist.all_gather_object(allocated_values, allocated)
    dist.all_gather_object(reserved_values, reserved)
    return tuple(allocated_values), tuple(reserved_values)


def train_distributed(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    context: DistributedContext,
    *,
    steps: int,
    batch_size_per_gpu: int,
    learning_rate: float,
    weight_decay: float,
    precision: str,
    seed: int,
) -> DistributedTrainResult:
    """Train with DDP and report global throughput and per-rank CUDA memory."""
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("precision must be 'float32' or 'bfloat16'")
    if precision == "bfloat16" and context.device.type != "cuda":
        precision = "float32"
    if steps < 1 or batch_size_per_gpu < 1:
        raise ValueError("steps and batch_size_per_gpu must be positive")

    model.to(context.device)
    dataset = TensorDataset(inputs, targets)
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        pin_memory=context.device.type == "cuda",
        num_workers=0,
    )
    wrapped: nn.Module = model
    if context.distributed:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)

    local_inputs, local_targets = shard_tensors(inputs, targets, context)
    wrapped.eval()
    with torch.no_grad():
        initial_loss = _global_mean(
            language_model_loss(wrapped, local_inputs, local_targets), context
        )

    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    tokens_processed = 0
    epoch = 0
    step = 0
    wrapped.train()
    while step < steps:
        sampler.set_epoch(epoch)
        for batch_inputs, batch_targets in loader:
            if step >= steps:
                break
            batch_inputs = batch_inputs.to(context.device, non_blocking=True)
            batch_targets = batch_targets.to(context.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=context.device.type,
                dtype=dtype,
                enabled=precision == "bfloat16",
            ):
                loss = language_model_loss(wrapped, batch_inputs, batch_targets)
            loss.backward()
            optimizer.step()
            tokens_processed += batch_targets.numel()
            step += 1
        epoch += 1

    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    elapsed = torch.tensor(time.perf_counter() - started, device=context.device)
    token_count = torch.tensor(float(tokens_processed), device=context.device)
    if context.distributed:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    wrapped.eval()
    with torch.no_grad():
        final_loss = _global_mean(
            language_model_loss(wrapped, local_inputs, local_targets), context
        )
    allocated, reserved = collect_peak_memory(context)
    return DistributedTrainResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        seconds=float(elapsed),
        tokens_per_second=float(token_count / elapsed),
        peak_allocated_bytes_per_rank=allocated,
        peak_reserved_bytes_per_rank=reserved,
        world_size=context.world_size,
        precision=precision,
    )
