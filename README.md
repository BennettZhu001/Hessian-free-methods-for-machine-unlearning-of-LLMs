# Tiny Transformer Unlearning

A small, auditable experiment in **Hessian-free approximate machine unlearning**.
The repository trains a one-block, single-head causal transformer, removes a
specified subset of training sequences, and compares three models:

1. the model trained on all data,
2. a one-step damped Newton unlearning update computed with Hessian-vector
   products (HVPs) and conjugate gradients (CG), and
3. a from-scratch retraining reference trained on the retained data.

The default experiment updates only the language-model head during unlearning.
This makes the retained-data objective convex conditional on the learned
transformer representation, so the second-order approximation is easy to
inspect. An `all` mode applies the same Hessian-free machinery to every model
parameter, but that experiment is non-convex and should be interpreted as a
local approximation rather than a certificate.

## What this demonstrates

- a complete data → training → deletion → unlearning → evaluation pipeline;
- a causal transformer implemented directly in PyTorch;
- exact autograd HVPs without constructing a Hessian matrix;
- a matrix-free CG solve for a damped Newton step;
- comparison with from-scratch retraining on retain loss, forget loss, predictive KL,
  parameter distance, runtime, and CG residuals.

This is a pedagogical research prototype, not a claim of certified removal or a
production-scale LLM unlearning system.

## Mathematical target

Let the retained empirical objective be

$$
F_R(\theta)=\frac{1}{|R|}\sum_{i\in R}\ell_i(\theta)
             + \frac{\lambda}{2}\|\theta\|_2^2.
$$

Starting from the full-data solution $\hat\theta$, the code computes

$$
(\nabla^2 F_R(\hat\theta)+\gamma I)\Delta
  = -\nabla F_R(\hat\theta)
$$

with CG. Every product $\nabla^2F_R(\hat\theta)v$ is evaluated by automatic
differentiation; the Hessian is never stored. The unlearned parameters are
$\hat\theta+\Delta$. See [`docs/method.md`](docs/method.md) for assumptions and
limitations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_experiment.py
```

The script prints a JSON summary and writes `artifacts/metrics.json`. A typical
CPU run uses a tiny synthetic sequence dataset and needs no network access.

## Reproduced default result

On the checked-in seed-7 configuration, the matrix-free head update took about
0.012 seconds, compared with about 0.6 seconds for from-scratch retained-data
training. It lowered retained NLL from 0.366 to 0.350; the retraining reference
reached 0.347. Deleted-sequence NLL rose from 0.216 to 0.411, whereas retraining
reached 13.037. Thus the update efficiently repairs retained-data fit but does
**not** reproduce the full forgetting behavior of retraining. The raw output is
in [`results/default_metrics.json`](results/default_metrics.json).

Useful options:

```bash
# More CG iterations and verbose diagnostics
python scripts/run_experiment.py --cg-max-iter 50 --verbose

# Non-convex, all-parameter local update
python scripts/run_experiment.py --unlearn-parameters all

# Reproduce exactly
python scripts/run_experiment.py --seed 7
```

## Repository layout

```text
src/tiny_unlearning/data.py       synthetic retained/forget data
src/tiny_unlearning/model.py      one-block single-head causal transformer
src/tiny_unlearning/training.py   deterministic full-batch optimization
src/tiny_unlearning/second_order.py  flattening, HVPs, and CG
src/tiny_unlearning/experiment.py end-to-end benchmark
scripts/run_experiment.py         command-line entry point
tests/                            numerical unit and integration tests
```

## Tests

```bash
pip install -e '.[test]'
pytest -q
```

The numerical tests check CG against a direct linear solve, HVPs against an
explicit Hessian on a quadratic, causal masking, and the end-to-end experiment.

## Scope and next steps

The deliberately small default isolates algorithmic behavior. Natural research
extensions are sequential deletion requests, preconditioned CG, warm starts,
low-rank curvature approximations, LoRA-restricted updates, stochastic HVPs,
and stronger empirical unlearning audits such as membership-inference and
exposure tests.

## Research boundary

The Newton/influence update itself is a standard baseline. This public
repository demonstrates the computational primitives underlying ongoing work;
it intentionally contains no claim of algorithmic novelty and no unpublished
theorem. The research questions begin with sequential deletions, accumulated
approximation error, scalable preconditioning, and auditable notions of model
discrepancy.

## Selected references

- Koh and Liang (2017), [Understanding Black-box Predictions via Influence
  Functions](https://proceedings.mlr.press/v70/koh17a.html).
- Guo et al. (2020), [Certified Data Removal from Machine Learning
  Models](https://proceedings.mlr.press/v119/guo20c.html).
- Bourtoule et al. (2021), [Machine Unlearning](https://doi.org/10.1109/SP40001.2021.00019).

## License

MIT. See [`LICENSE`](LICENSE).
