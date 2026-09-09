from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM


@dataclass
class VLMConfig(MiniMindConfig):
    image_hidden_size: int = 768
    image_token_id: int = 12
    image_token_length: int = 64


class VisionProjector(nn.Module):
    def __init__(self, image_hidden_size: int, language_hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(image_hidden_size),
            nn.Linear(image_hidden_size, language_hidden_size),
            nn.GELU(),
            nn.Linear(language_hidden_size, language_hidden_size),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.layers(image_features)


class MiniMindVLM(nn.Module):
    """LLaVA-style early-fusion VLM with a frozen vision encoder."""

    def __init__(self, config: VLMConfig, vision_encoder: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.language_model = MiniMindForCausalLM(config)
        self.vision_encoder = vision_encoder.eval()
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        self.vision_projector = VisionProjector(config.image_hidden_size, config.hidden_size)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.vision_encoder(pixel_values)
        features = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        if features.ndim != 3:
            raise ValueError("vision encoder must return [batch, image_tokens, image_hidden]")
        return self.vision_projector(features)

    def inject_image_features(
        self, input_ids: torch.Tensor, text_embeddings: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        output = []
        for batch_index in range(input_ids.size(0)):
            positions = torch.nonzero(input_ids[batch_index] == self.config.image_token_id).flatten()
            if positions.numel() != self.config.image_token_length:
                raise ValueError(
                    f"sample {batch_index} has {positions.numel()} image placeholders; "
                    f"expected {self.config.image_token_length}"
                )
            if image_features.size(1) != self.config.image_token_length:
                raise ValueError("vision encoder token count does not match image_token_length")
            embeddings = text_embeddings[batch_index].clone()
            embeddings[positions] = image_features[batch_index]
            output.append(embeddings)
        return torch.stack(output)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        text_embeddings = self.language_model.embed_tokens(input_ids)
        image_features = self.encode_images(pixel_values)
        inputs_embeds = self.inject_image_features(input_ids, text_embeddings, image_features)
        return self.language_model.forward_from_embeddings(inputs_embeds, labels)

    def set_alignment_trainable(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.vision_projector.parameters():
            parameter.requires_grad = True

    def set_instruction_trainable(self) -> None:
        self.set_alignment_trainable()
        for layer_index in (0, self.config.num_hidden_layers - 1):
            for parameter in self.language_model.layers[layer_index].parameters():
                parameter.requires_grad = True
