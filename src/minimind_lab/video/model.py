from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM
from minimind_lab.vlm.model import VisionProjector


@dataclass
class VideoOmniConfig(MiniMindConfig):
    vision_hidden_size: int = 768
    video_token_id: int = 13
    num_frames: int = 8
    num_video_tokens: int = 16
    temporal_layers: int = 2
    temporal_heads: int = 8


class TemporalVideoAdapter(nn.Module):
    """Order-aware temporal encoder followed by learned-query resampling."""

    def __init__(self, config: VideoOmniConfig) -> None:
        super().__init__()
        self.frame_positions = nn.Parameter(torch.empty(config.num_frames, config.vision_hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=config.vision_hidden_size,
            nhead=config.temporal_heads,
            dim_feedforward=config.vision_hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=config.temporal_layers, enable_nested_tensor=False
        )
        self.queries = nn.Parameter(torch.empty(config.num_video_tokens, config.vision_hidden_size))
        self.resampler = nn.MultiheadAttention(
            config.vision_hidden_size, config.temporal_heads, dropout=config.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(config.vision_hidden_size)
        nn.init.normal_(self.frame_positions, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError("frame features must be [batch, frames, hidden]")
        frame_count = frame_features.size(1)
        if frame_count > self.frame_positions.size(0):
            raise ValueError("video has more frames than configured")
        temporal = self.temporal_encoder(frame_features + self.frame_positions[:frame_count])
        queries = self.queries.unsqueeze(0).expand(frame_features.size(0), -1, -1)
        resampled, _ = self.resampler(queries, temporal, temporal, need_weights=False)
        return self.norm(resampled)


class MiniMindVideoOmni(nn.Module):
    """Video-to-text model: frozen frame encoder, temporal adapter, and MiniMind LLM."""

    def __init__(self, config: VideoOmniConfig, vision_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.language_model = MiniMindForCausalLM(config)
        self.vision_encoder = vision_encoder.eval()
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        self.temporal_adapter = TemporalVideoAdapter(config)
        self.video_projector = VisionProjector(config.vision_hidden_size, config.hidden_size)

    def encode_video(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError("pixel values must be [batch, frames, channels, height, width]")
        batch_size, frame_count = pixel_values.shape[:2]
        with torch.no_grad():
            output = self.vision_encoder(pixel_values.flatten(0, 1))
        features = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        if features.ndim != 3:
            raise ValueError("vision encoder must return [batch*frames, patches, hidden]")
        frame_features = features.mean(dim=1).view(batch_size, frame_count, -1)
        return self.video_projector(self.temporal_adapter(frame_features))

    def inject_video_features(
        self, input_ids: torch.Tensor, text_embeddings: torch.Tensor, video_features: torch.Tensor
    ) -> torch.Tensor:
        output = []
        for batch_index in range(input_ids.size(0)):
            positions = torch.nonzero(input_ids[batch_index] == self.config.video_token_id).flatten()
            if positions.numel() != self.config.num_video_tokens:
                raise ValueError(
                    f"sample {batch_index} has {positions.numel()} video placeholders; "
                    f"expected {self.config.num_video_tokens}"
                )
            embeddings = text_embeddings[batch_index].clone()
            embeddings[positions] = video_features[batch_index]
            output.append(embeddings)
        return torch.stack(output)

    def forward_with_features(
        self, input_ids: torch.Tensor, video_features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        embeddings = self.language_model.embed_tokens(input_ids)
        embeddings = self.inject_video_features(input_ids, embeddings, video_features)
        return self.language_model.forward_from_embeddings(embeddings, labels)

    def forward(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        return self.forward_with_features(input_ids, self.encode_video(pixel_values), labels)

    def set_alignment_trainable(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.temporal_adapter, self.video_projector):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def set_instruction_trainable(self) -> None:
        self.set_alignment_trainable()
        for layer_index in (0, self.config.num_hidden_layers - 1):
            for parameter in self.language_model.layers[layer_index].parameters():
                parameter.requires_grad = True

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        video_features = self.encode_video(pixel_values)
        for _ in range(max_new_tokens):
            logits = self.forward_with_features(input_ids, video_features)["logits"][:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return input_ids
