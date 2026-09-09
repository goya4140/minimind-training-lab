from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minimind_lab.video import MiniMindVideoOmni, VideoOmniConfig


class FakeVision(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.projection = nn.Linear(3, hidden_size)

    def forward(self, pixels):
        pooled = pixels.mean(dim=(-1, -2))
        tokens = self.projection(pooled).unsqueeze(1).expand(-1, 4, -1)
        return SimpleNamespace(last_hidden_state=tokens)


def config() -> VideoOmniConfig:
    return VideoOmniConfig(
        vocab_size=259,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        vision_hidden_size=32,
        video_token_id=13,
        num_frames=4,
        num_video_tokens=3,
        temporal_layers=1,
        temporal_heads=4,
    )


def test_video_omni_forward_and_loss_are_finite():
    model = MiniMindVideoOmni(config(), FakeVision())
    input_ids = torch.randint(20, 259, (2, 12))
    input_ids[:, 1:4] = 13
    labels = input_ids.clone()
    labels[:, :5] = -100
    output = model(input_ids, torch.randn(2, 4, 3, 8, 8), labels)
    assert output["logits"].shape == (2, 12, 259)
    assert torch.isfinite(output["loss"])


def test_video_adapter_preserves_frame_order_information():
    torch.manual_seed(7)
    model = MiniMindVideoOmni(config(), FakeVision())
    pixels = torch.arange(2 * 4 * 3 * 2 * 2, dtype=torch.float32).view(2, 4, 3, 2, 2)
    normal = model.encode_video(pixels)
    reversed_features = model.encode_video(pixels.flip(1))
    assert not torch.allclose(normal, reversed_features)


def test_alignment_and_instruction_freeze_policies():
    model = MiniMindVideoOmni(config(), FakeVision())
    model.set_alignment_trainable()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith(("temporal_adapter.", "video_projector.")) for name in trainable)

    model.set_instruction_trainable()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any(name.startswith("language_model.layers.0.") for name in trainable)
    assert any(name.startswith("language_model.layers.3.") for name in trainable)
    assert not any(name.startswith("vision_encoder.") for name in trainable)


def test_video_placeholder_count_is_strict():
    model = MiniMindVideoOmni(config(), FakeVision())
    input_ids = torch.tensor([[13, 13, 20]])
    with pytest.raises(ValueError, match="video placeholders"):
        model.inject_video_features(input_ids, model.language_model.embed_tokens(input_ids), torch.randn(1, 3, 64))
