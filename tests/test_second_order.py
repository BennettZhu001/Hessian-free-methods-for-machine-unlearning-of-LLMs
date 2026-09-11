import torch

from tiny_unlearning.second_order import conjugate_gradient, flatten


def test_conjugate_gradient_matches_direct_solve() -> None:
    matrix = torch.tensor(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    result = conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        max_iter=10,
        relative_tolerance=1e-12,
    )
    expected = torch.linalg.solve(matrix, rhs)
    assert result.converged
    assert torch.allclose(result.solution, expected, atol=1e-10, rtol=1e-10)


def test_autograd_hvp_matches_quadratic_hessian() -> None:
    matrix = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    parameter = torch.tensor([0.2, -0.7], dtype=torch.float64, requires_grad=True)
    vector = torch.tensor([1.3, -0.4], dtype=torch.float64)
    objective = 0.5 * parameter @ matrix @ parameter
    gradient = torch.autograd.grad(objective, [parameter], create_graph=True)
    flat_gradient = flatten(gradient)
    hvp = torch.autograd.grad(flat_gradient, [parameter], grad_outputs=vector)[0]
    assert torch.allclose(hvp, matrix @ vector, atol=1e-12, rtol=1e-12)
