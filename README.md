# Hessian-free sequential transformer unlearning

An auditable progression from a tiny reference experiment to distributed,
parameter-efficient **Hessian-free approximate machine unlearning**.

The repository now contains two deliberately separate tracks:

- **Tiny reference:** a one-block, single-head transformer for inspecting every
  Hessian-vector product and comparing directly with retraining.
- **1M milestone:** a 1,165,504-parameter, four-layer transformer; DDP training;
  sequential deletion requests; a rank-eight LoRA update; distributed HVPs;
  memory and throughput instrumentation; and a recovery fine-tuning audit.

The 1M path is implemented and CPU-smoke-testable. No RTX 5090 measurements are
claimed until the checked-in two-GPU configuration is run on that hardware and
its raw output is reviewed. This remains a research prototype, not a claim of
certified removal or a production-scale LLM unlearning system.

## 1M parameter milestone on two GPUs

Install the package and run the control-flow smoke test:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -e '.[test]'
python scripts/run_scalable_experiment.py --config configs/cpu_smoke.json
```

On a machine whose CUDA devices 0 and 1 are the two RTX 5090s:

```bash
bash scripts/launch_2x5090.sh
```

To run a matched-global-batch one-versus-two GPU scaling comparison:

```bash
bash scripts/benchmark_1v2_5090.sh
```

The launcher performs a hardware preflight and then uses:

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/run_scalable_experiment.py \
  --config configs/1m_2x5090.json \
  --require-world-size 2
```

The output records exact GPU identities, PyTorch version, tokens/second,
per-rank peak CUDA memory, every CG residual trace, predictive KL to final
retraining, and the result of an adapter recovery attack. See
[`docs/scaling.md`](docs/scaling.md) for design choices and limitations.

## Tiny reference experiment

The original reference trains a one-block, single-head causal transformer,
removes a specified subset of training sequences, and compares three models:

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
src/tiny_unlearning/scalable_model.py  configurable multi-head transformer + LoRA
src/tiny_unlearning/sequential_data.py sequential deletion-request corpus
src/tiny_unlearning/distributed.py     DDP, distributed HVPs, and profiling
src/tiny_unlearning/scalable_experiment.py  1M end-to-end experiment
scripts/run_experiment.py         command-line entry point
scripts/run_scalable_experiment.py  single- or multi-process scale entry point
scripts/benchmark_1v2_5090.sh     matched-global-batch scaling benchmark
configs/                          CPU smoke and 1x/2x RTX 5090 configurations
tests/                            numerical unit and integration tests
```

## Tests

```bash
pip install -e '.[test]'
pytest -q
```

The numerical tests check CG against a direct linear solve, HVPs against an
explicit Hessian on a quadratic, causal masking, LoRA initialization and
parameter selection, sequential dataset semantics, both end-to-end paths, and
the parameter count of the 1M preset. GitHub Actions runs tests and Ruff checks.

## Scope and next steps

The next scale stage is not obtained by changing a parameter-count flag. A 10B+
experiment requires FSDP or ZeRO-3, an external tokenized dataset, activation
checkpointing, checkpoint/resume support, and stronger audits such as
membership inference and extraction attacks. Those features are roadmap items,
not present-tense claims.

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
