from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch


class SenseVoiceProcessor:
    def __init__(self, frontend) -> None:
        self.frontend = frontend.eval()

    def __call__(self, waveform, **_kwargs):
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            features, lengths = self.frontend(waveform, torch.tensor([waveform.size(1)]))
        mask = torch.arange(features.size(1)) < lengths[0]
        return SimpleNamespace(input_features=features, attention_mask=mask.unsqueeze(0).long())


def load_sensevoice(path: str | Path, device: torch.device):
    from funasr import AutoModel

    bundle = AutoModel(model=str(path), trust_remote_code=True, disable_update=True, device="cpu")
    encoder = bundle.model.encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, SenseVoiceProcessor(bundle.kwargs["frontend"])
