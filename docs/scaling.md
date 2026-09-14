# 1M-parameter distributed milestone

This milestone separates three claims that are often conflated:

1. **Distributed training:** the full-data model and the final retained-data
   retraining reference use PyTorch DistributedDataParallel (DDP).
2. **Parameter-efficient unlearning:** the transformer is frozen and each
   sequential update changes only the up projection of a LoRA language-model
   head.
3. **Distributed second-order computation:** each rank computes a local
   Hessian-vector product (HVP); the products are averaged before every
   conjugate-gradient (CG) iteration. No dense Hessian is materialized.

The default model has 1,165,504 base parameters before the LoRA adapter:

- vocabulary 256 and context length 64;
- width 160, four attention heads, and four transformer blocks;
- feed-forward width 512;
- rank-eight LoRA head for the unlearning updates.

The synthetic corpus contains modular background sequences and four distinct
canary groups. The groups become deletion requests one at a time. After each
request, the code records retained and cumulatively deleted NLL, CG convergence,
and runtime. At the end it compares with a model retrained from initialization
on the final retained dataset.

## Installation and preflight

Install a PyTorch build compatible with the machine's NVIDIA driver first,
then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e '.[test]'
python scripts/check_hardware.py
```

Verify that the preflight output lists exactly the intended RTX 5090 devices.
A standard RTX 4090 has 24 GiB; if a machine reports 48 GiB, preserve the exact
device name and memory output rather than describing it from memory.

## CPU smoke test

```bash
python scripts/run_scalable_experiment.py \
  --config configs/cpu_smoke.json \
  --output artifacts/cpu_smoke_metrics.json
```

This validates control flow only. It is not a performance result.

## Two-GPU run

```bash
bash scripts/launch_2x5090.sh
```

The launcher requires `WORLD_SIZE=2`; it fails rather than silently running on
one GPU. Raw results are written to `artifacts/1m_2x5090_metrics.json`, which is
ignored by Git so that an unexecuted configuration cannot be mistaken for a
reproduced result.

## One-versus-two GPU benchmark

```bash
bash scripts/benchmark_1v2_5090.sh
```

This runs the same architecture and global batch size first on one 5090 and
then on two. The one-GPU configuration uses 64 sequences per step; each rank in
the two-GPU configuration uses 32. The comparison script reports

$$
\text{speedup}=\frac{\text{tokens/s on two GPUs}}{\text{tokens/s on one GPU}},
\qquad
\text{efficiency}=\frac{\text{speedup}}{2}.
$$

This is a strong-scaling measurement for the training phase. The unlearning
phase also differs in how its fixed global curvature batch is divided across
ranks, but its runtime is reported separately and is not used in the training
scaling calculation.

## What the metrics do and do not establish

The result records device identities, PyTorch version, per-rank peak allocated
and reserved memory, training throughput, per-request HVP/CG diagnostics,
predictive KL to retraining, and a recovery fine-tuning attack.

The recovery attack asks whether a small number of adapter-only gradient steps
can restore low loss on the deleted canaries. It is deliberately simple. A low
recovery score against this attack is not a robustness certificate, and canary
forgetting is not proof that arbitrary training data were removed.

DDP replicates the 1M-parameter model and therefore demonstrates distributed
execution and throughput, not larger model capacity. Moving toward 10B models
will require sharded parameters and optimizer state (FSDP or ZeRO-3), external
tokenized datasets, activation checkpointing, and more realistic attacks.
