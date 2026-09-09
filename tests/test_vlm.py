from types import SimpleNamespace

import torch
from torch import nn

from minimind_lab.vlm import MiniMindVLM, VLMConfig


class FakeVisionEncoder(nn.Module):
    def __init__(self, tokens: int, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, tokens * hidden)
        self.tokens = tokens
        self.hidden = hidden

    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(-1, -2))
        return SimpleNamespace(last_hidden_state=self.projection(pooled).view(-1, self.tokens, self.hidden))


def config() -> VLMConfig:
    return VLMConfig(
        vocab_size=259,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
        image_hidden_size=48,
        image_token_id=12,
        image_token_length=4,
    )


def test_vlm_forward_and_frozen_encoder():
    model = MiniMindVLM(config(), FakeVisionEncoder(tokens=4, hidden=48))
    input_ids = torch.randint(20, 259, (2, 16))
    input_ids[:, 3:7] = 12
    output = model(input_ids, torch.randn(2, 3, 8, 8), labels=input_ids)
    assert output["logits"].shape == (2, 16, 259)
    assert torch.isfinite(output["loss"])
    assert not any(parameter.requires_grad for parameter in model.vision_encoder.parameters())


def test_alignment_freezes_everything_except_projector():
    model = MiniMindVLM(config(), FakeVisionEncoder(tokens=4, hidden=48))
    model.set_alignment_trainable()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("vision_projector.") for name in trainable)


def test_instruction_stage_unfreezes_boundary_layers():
    model = MiniMindVLM(config(), FakeVisionEncoder(tokens=4, hidden=48))
    model.set_instruction_trainable()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert any(name.startswith("vision_projector.") for name in trainable)
    assert any(name.startswith("language_model.layers.0.") for name in trainable)
    assert any(name.startswith("language_model.layers.1.") for name in trainable)

