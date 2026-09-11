from tiny_unlearning import ExperimentConfig, run_experiment


def test_end_to_end_head_unlearning_reduces_retained_loss() -> None:
    metrics = run_experiment(
        ExperimentConfig(
            n_retain=24,
            n_forget=4,
            training_steps=20,
            d_model=8,
            d_ff=16,
            cg_max_iter=10,
            damping=0.1,
        )
    )
    full_loss = metrics["losses"]["full_data_model"]["retain_nll"]
    unlearned_loss = metrics["losses"]["newton_unlearned"]["retain_nll"]
    assert unlearned_loss <= full_loss + 1e-6
    assert (
        metrics["cg"]["cg_residual_norms"][0] >= metrics["cg"]["cg_residual_norms"][-1]
    )


def test_all_parameter_hvp_path_supports_double_backward() -> None:
    metrics = run_experiment(
        ExperimentConfig(
            n_retain=12,
            n_forget=2,
            training_steps=3,
            d_model=4,
            d_ff=8,
            cg_max_iter=2,
            damping=2.0,
            unlearn_parameters="all",
        )
    )
    assert metrics["model"]["updated_fraction"] == 1.0
    assert metrics["cg"]["cg_iterations"] >= 1
