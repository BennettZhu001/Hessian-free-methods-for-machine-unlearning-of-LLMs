import torch

from tiny_unlearning.model import TinyCausalTransformer


def test_causal_mask_blocks_future_tokens() -> None:
    torch.manual_seed(0)
    model = TinyCausalTransformer(vocab_size=13, context_length=6)
    model.eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 9, 10, 11]])
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    assert torch.allclose(
        first_logits[:, :3], second_logits[:, :3], atol=1e-6, rtol=1e-6
    )
