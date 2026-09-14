import torch

from tiny_unlearning.scalable_model import (
    FixedDownLoRAHead,
    ScalableCausalTransformer,
    TransformerConfig,
)
from tiny_unlearning.sequential_data import make_sequential_corpus


def test_one_million_preset_has_expected_scale() -> None:
    model = ScalableCausalTransformer(TransformerConfig())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 1_000_000 <= parameters <= 1_300_000


def test_lora_head_is_zero_initialized_and_parameter_efficient() -> None:
    torch.manual_seed(0)
    model = ScalableCausalTransformer(
        TransformerConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
        )
    )
    token_ids = torch.randint(0, 32, (2, 8))
    model.eval()
    with torch.no_grad():
        before = model(token_ids)
    model.enable_lora_head(rank=2, alpha=2.0, seed=3)
    with torch.no_grad():
        after = model(token_ids)
    assert isinstance(model.lm_head, FixedDownLoRAHead)
    assert torch.equal(before, after)
    assert sum(p.numel() for p in model.trainable_parameters()) == 32 * 2
    assert not model.lm_head.lora_a.requires_grad
    assert model.lm_head.lora_b.requires_grad


def test_scalable_attention_is_causal() -> None:
    torch.manual_seed(0)
    model = ScalableCausalTransformer(
        TransformerConfig(
            vocab_size=32,
            context_length=6,
            d_model=16,
            n_heads=2,
            n_layers=2,
            d_ff=32,
        )
    )
    model.eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 9, 10, 11]])
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    assert torch.allclose(
        first_logits[:, :3], second_logits[:, :3], atol=1e-6, rtol=1e-6
    )


def test_sequential_corpus_removes_one_request_at_a_time() -> None:
    corpus = make_sequential_corpus(
        n_retain=10,
        n_requests=3,
        examples_per_request=2,
        sequence_length=9,
        vocab_size=32,
    )
    assert corpus.full_inputs.shape == (16, 8)
    assert corpus.retained_after(0)[0].shape[0] == 16
    assert corpus.retained_after(1)[0].shape[0] == 14
    assert corpus.retained_after(3)[0].shape[0] == 10
    assert corpus.deleted_through(2)[0].shape[0] == 4
