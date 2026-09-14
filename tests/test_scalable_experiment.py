from tiny_unlearning.distributed import initialize_distributed
from tiny_unlearning.scalable_experiment import (
    ScalableExperimentConfig,
    run_scalable_experiment,
)


def test_cpu_smoke_experiment_runs_full_pipeline() -> None:
    context = initialize_distributed()
    metrics = run_scalable_experiment(
        ScalableExperimentConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            n_retain=12,
            n_requests=2,
            examples_per_request=2,
            training_steps=2,
            batch_size_per_gpu=4,
            learning_rate=1e-3,
            precision="float32",
            lora_rank=2,
            lora_alpha=2.0,
            curvature_examples_per_rank=4,
            damping=0.1,
            cg_max_iter=2,
            recovery_steps=1,
            train_retraining_reference=True,
        ),
        context,
    )
    assert metrics is not None
    assert len(metrics["sequential_unlearning"]["requests"]) == 2
    assert metrics["model"]["trainable_parameters"] == 64
    assert metrics["comparison_to_final_retraining"] is not None
    assert "nll_recovery" in metrics["robustness_audit"]
