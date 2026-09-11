"""Matrix-free Hessian-vector products and conjugate gradients."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

TensorVectorProduct = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CGResult:
    solution: torch.Tensor
    iterations: int
    residual_norms: list[float]
    converged: bool
    breakdown: bool


@dataclass(frozen=True)
class NewtonResult:
    gradient_norm: float
    step_norm: float
    cg_iterations: int
    cg_residual_norms: list[float]
    cg_converged: bool
    cg_breakdown: bool


def flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    pieces = [tensor.reshape(-1) for tensor in tensors]
    if not pieces:
        raise ValueError("at least one tensor is required")
    return torch.cat(pieces)


def unflatten_like(
    vector: torch.Tensor, tensors: Iterable[torch.Tensor]
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    offset = 0
    for tensor in tensors:
        size = tensor.numel()
        outputs.append(vector[offset : offset + size].view_as(tensor))
        offset += size
    if offset != vector.numel():
        raise ValueError("vector length does not match reference tensors")
    return outputs


def conjugate_gradient(
    matvec: TensorVectorProduct,
    rhs: torch.Tensor,
    *,
    max_iter: int = 30,
    relative_tolerance: float = 1e-7,
    absolute_tolerance: float = 1e-10,
) -> CGResult:
    """Solve Ax=b using CG, assuming a symmetric positive-definite operator."""
    if rhs.ndim != 1:
        raise ValueError("rhs must be a flat vector")
    solution = torch.zeros_like(rhs)
    residual = rhs - matvec(solution)
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    initial_norm = float(torch.sqrt(residual_squared))
    residual_norms = [initial_norm]
    threshold = max(absolute_tolerance, relative_tolerance * initial_norm)
    if initial_norm <= threshold:
        return CGResult(solution, 0, residual_norms, True, False)

    for iteration in range(1, max_iter + 1):
        operator_direction = matvec(direction)
        curvature = torch.dot(direction, operator_direction)
        if not torch.isfinite(curvature) or float(curvature) <= 0.0:
            return CGResult(solution, iteration - 1, residual_norms, False, True)
        step_size = residual_squared / curvature
        solution = solution + step_size * direction
        new_residual = residual - step_size * operator_direction
        new_residual_squared = torch.dot(new_residual, new_residual)
        norm = float(torch.sqrt(torch.clamp_min(new_residual_squared, 0.0)))
        residual_norms.append(norm)
        if norm <= threshold:
            return CGResult(solution, iteration, residual_norms, True, False)
        beta = new_residual_squared / residual_squared
        direction = new_residual + beta * direction
        residual = new_residual
        residual_squared = new_residual_squared

    return CGResult(solution, max_iter, residual_norms, False, False)


def selected_parameters(model: nn.Module, mode: str) -> list[nn.Parameter]:
    if mode == "head":
        return list(model.lm_head.parameters())
    if mode == "all":
        return list(model.parameters())
    raise ValueError("mode must be 'head' or 'all'")


def damped_newton_step(
    objective: Callable[[], torch.Tensor],
    parameters: list[nn.Parameter],
    *,
    damping: float,
    cg_max_iter: int,
    cg_relative_tolerance: float,
) -> NewtonResult:
    """Take one matrix-free damped Newton step on ``objective``.

    The gradient graph is constructed once. CG repeatedly queries it through
    vector-Jacobian products; no dense Hessian is formed.
    """
    if damping < 0:
        raise ValueError("damping must be nonnegative")
    loss = objective()
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    flat_gradient = flatten(gradients)
    rhs = -flat_gradient.detach()

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        products = torch.autograd.grad(
            flat_gradient,
            parameters,
            grad_outputs=vector,
            retain_graph=True,
        )
        return flatten(products).detach() + damping * vector

    cg = conjugate_gradient(
        matvec,
        rhs,
        max_iter=cg_max_iter,
        relative_tolerance=cg_relative_tolerance,
    )
    step_tensors = unflatten_like(cg.solution, parameters)
    with torch.no_grad():
        for parameter, step in zip(parameters, step_tensors, strict=True):
            parameter.add_(step)

    return NewtonResult(
        gradient_norm=float(torch.linalg.vector_norm(rhs)),
        step_norm=float(torch.linalg.vector_norm(cg.solution)),
        cg_iterations=cg.iterations,
        cg_residual_norms=cg.residual_norms,
        cg_converged=cg.converged,
        cg_breakdown=cg.breakdown,
    )
