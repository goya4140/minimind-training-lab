from types import SimpleNamespace

import torch
from torch import nn

from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
from minimind_lab.video import MiniMindVideoOmni, VideoOmniConfig
from minimind_lab.vlm import MiniMindVLM, VLMConfig


class ParameterFreeVision(nn.Module):
    def __init__(self, tokens: int, hidden: int) -> None:
        super().__init__()
        self.tokens = tokens
        self.hidden = hidden

    def forward(self, pixels):
        return SimpleNamespace(
            last_hidden_state=torch.zeros(pixels.size(0), self.tokens, self.hidden, device=pixels.device)
        )


def test_formal_llm_parameter_count_matches_handbook():
    assert MiniMindForCausalLM(MiniMindConfig()).parameter_count() == 63_912_192


def test_formal_vlm_projector_parameter_count_matches_handbook():
    model = MiniMindVLM(VLMConfig(), ParameterFreeVision(tokens=64, hidden=768))
    projector = sum(parameter.numel() for parameter in model.vision_projector.parameters())
    assert projector == 1_182_720
    model.set_alignment_trainable()
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 1_182_720


def test_formal_video_trainable_counts_match_handbook():
    model = MiniMindVideoOmni(VideoOmniConfig(), ParameterFreeVision(tokens=64, hidden=768))
    assert sum(parameter.numel() for parameter in model.parameters()) == 84_017_664
    model.set_alignment_trainable()
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 20_105_472
    model.set_instruction_trainable()
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 34_854_528
