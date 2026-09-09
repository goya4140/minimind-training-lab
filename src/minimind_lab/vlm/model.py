from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
        num_images = 1
        if pixel_values.ndim == 5:
            batch_size, num_images = pixel_values.shape[:2]
            pixel_values = pixel_values.flatten(0, 1)
        with torch.no_grad():
            output = self.vision_encoder(pixel_values)
        features = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        if features.ndim != 3:
            raise ValueError("vision encoder must return [batch, image_tokens, image_hidden]")
        projected = self.vision_projector(features)
        if num_images > 1:
            projected = projected.view(batch_size, num_images, projected.size(1), projected.size(2))
        return projected

    def inject_image_features(
        self,
        input_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_features: torch.Tensor,
        image_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_features.ndim == 3:
            image_features = image_features.unsqueeze(1)
        output = []
        for batch_index in range(input_ids.size(0)):
            count = int(image_counts[batch_index]) if image_counts is not None else image_features.size(1)
            positions = torch.nonzero(input_ids[batch_index] == self.config.image_token_id).flatten()
            expected = count * self.config.image_token_length
            if positions.numel() != expected:
                raise ValueError(
                    f"sample {batch_index} has {positions.numel()} image placeholders; "
                    f"expected {expected} for {count} image(s)"
                )
            if image_features.size(2) != self.config.image_token_length:
                raise ValueError("vision encoder token count does not match image_token_length")
            embeddings = text_embeddings[batch_index].clone()
            embeddings[positions] = image_features[batch_index, :count].flatten(0, 1)
            output.append(embeddings)
        return torch.stack(output)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        image_counts: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        text_embeddings = self.language_model.embed_tokens(input_ids)
        image_features = self.encode_images(pixel_values)
        inputs_embeds = self.inject_image_features(input_ids, text_embeddings, image_features, image_counts)
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

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_counts: torch.Tensor | None = None,
        max_new_tokens: int = 80,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_position_embeddings :]
            logits = self(context, pixel_values, image_counts=image_counts)["logits"][:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return input_ids
