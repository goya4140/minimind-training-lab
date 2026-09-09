from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from minimind_lab.llm.model import DecoderBlock, MiniMindConfig, MiniMindForCausalLM, RMSNorm
from minimind_lab.vlm.model import VisionProjector


@dataclass
class OmniConfig(MiniMindConfig):
    num_talker_hidden_layers: int = 4
    talker_hidden_size: int = 768
    num_audio_codebooks: int = 8
    audio_vocab_size: int = 2112
    audio_hidden_size: int = 512
    audio_token_id: int = 16
    image_hidden_size: int = 768
    image_token_id: int = 12
    image_token_length: int = 64
    speaker_embedding_size: int = 192
    audio_speaker_token_id: int = 2051
    bridge_layer: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.talker_hidden_size % self.num_attention_heads:
            raise ValueError("talker_hidden_size must be divisible by num_attention_heads")
        if self.bridge_layer is None:
            self.bridge_layer = self.num_hidden_layers // 2 - 1


@dataclass
class OmniOutput:
    text_logits: torch.Tensor
    audio_logits: list[torch.Tensor]
    text_loss: torch.Tensor | None = None
    audio_loss: torch.Tensor | None = None

    @property
    def loss(self) -> torch.Tensor | None:
        if self.text_loss is None:
            return self.audio_loss
        if self.audio_loss is None:
            return self.text_loss
        return self.text_loss + self.audio_loss


class AudioProjector(nn.Module):
    def __init__(self, audio_hidden_size: int, language_hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(audio_hidden_size),
            nn.Linear(audio_hidden_size, language_hidden_size),
            nn.GELU(),
            nn.Linear(language_hidden_size, language_hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class MultiCodebookEmbedding(nn.Module):
    def __init__(self, config: OmniConfig, adapter_rank: int = 256) -> None:
        super().__init__()
        self.base = nn.Embedding(config.audio_vocab_size, config.talker_hidden_size)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Embedding(config.audio_vocab_size, adapter_rank),
                nn.GELU(),
                nn.Linear(adapter_rank, config.talker_hidden_size, bias=False),
            )
            for _ in range(config.num_audio_codebooks)
        )

    def forward(self, code_ids: torch.Tensor) -> torch.Tensor:
        streams = [
            self.base(code_ids[:, index]) + adapter(code_ids[:, index]) for index, adapter in enumerate(self.adapters)
        ]
        return torch.stack(streams).mean(0)


class MultiCodebookHead(nn.Module):
    def __init__(self, config: OmniConfig, adapter_rank: int = 256) -> None:
        super().__init__()
        self.base = nn.Linear(config.talker_hidden_size, config.audio_vocab_size, bias=False)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(config.talker_hidden_size, adapter_rank, bias=False),
                nn.GELU(),
                nn.Linear(adapter_rank, config.audio_vocab_size, bias=False),
            )
            for _ in range(config.num_audio_codebooks)
        )

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        base = self.base(hidden)
        return [base + adapter(hidden) for adapter in self.adapters]


class Talker(nn.Module):
    def __init__(self, config: OmniConfig) -> None:
        super().__init__()
        talker_config = MiniMindConfig(
            vocab_size=config.audio_vocab_size,
            hidden_size=config.talker_hidden_size,
            num_hidden_layers=config.num_talker_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.code_embeddings = MultiCodebookEmbedding(config)
        self.semantic_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.talker_hidden_size),
            nn.GELU(),
            RMSNorm(config.talker_hidden_size, config.rms_norm_eps),
        )
        self.codec_projector = nn.Sequential(
            nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
            nn.GELU(),
            RMSNorm(config.talker_hidden_size, config.rms_norm_eps),
        )
        self.speaker_projector = nn.Linear(config.speaker_embedding_size, config.talker_hidden_size, bias=False)
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        self.layers = nn.ModuleList(DecoderBlock(talker_config) for _ in range(config.num_talker_hidden_layers))
        self.norm = RMSNorm(config.talker_hidden_size, config.rms_norm_eps)
        self.head = MultiCodebookHead(config)

    def forward(
        self,
        bridge_states: torch.Tensor,
        audio_code_ids: torch.Tensor,
        speaker_embedding: torch.Tensor | None = None,
        speaker_positions: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        audio_embeddings = self.code_embeddings(audio_code_ids)
        if speaker_embedding is not None and speaker_positions is not None:
            replacement = self.speaker_projector(speaker_embedding)
            audio_embeddings = audio_embeddings.clone()
            for batch_index, position in enumerate(speaker_positions.tolist()):
                audio_embeddings[batch_index, position] = replacement[batch_index]
        hidden = self.semantic_projector(bridge_states) * self.text_scale
        hidden = hidden + self.codec_projector(audio_embeddings) * self.audio_scale
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(self.norm(hidden))


class MiniMindOmni(nn.Module):
    """Trainable Thinker–Bridge–Talker core; external encoders/codecs stay outside."""

    def __init__(self, config: OmniConfig) -> None:
        super().__init__()
        self.config = config
        self.thinker = MiniMindForCausalLM(config)
        self.audio_projector = AudioProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_projector = VisionProjector(config.image_hidden_size, config.hidden_size)
        self.talker = Talker(config)

    @staticmethod
    def inject_features(
        token_ids: torch.Tensor, embeddings: torch.Tensor, features: torch.Tensor, placeholder_id: int
    ) -> torch.Tensor:
        output = []
        for batch_index in range(token_ids.size(0)):
            positions = torch.nonzero(token_ids[batch_index] == placeholder_id).flatten()
            if positions.numel() < features.size(1):
                raise ValueError("not enough modality placeholder tokens")
            item = embeddings[batch_index].clone()
            item[positions[: features.size(1)]] = features[batch_index]
            output.append(item)
        return torch.stack(output)

    def initialize_talker_from_thinker(self) -> None:
        if self.config.talker_hidden_size != self.config.hidden_size:
            raise ValueError("layer-copy initialization requires matching hidden sizes")
        source_start = len(self.thinker.layers) - len(self.talker.layers)
        for index, layer in enumerate(self.talker.layers):
            layer.load_state_dict(self.thinker.layers[source_start + index].state_dict())

    def forward(
        self,
        text_ids: torch.Tensor,
        audio_code_ids: torch.Tensor,
        text_labels: torch.Tensor | None = None,
        audio_labels: torch.Tensor | None = None,
        encoded_audio: torch.Tensor | None = None,
        encoded_images: torch.Tensor | None = None,
        speaker_embedding: torch.Tensor | None = None,
        speaker_positions: torch.Tensor | None = None,
    ) -> OmniOutput:
        hidden = self.thinker.embed_tokens(text_ids)
        if encoded_audio is not None:
            hidden = self.inject_features(
                text_ids, hidden, self.audio_projector(encoded_audio), self.config.audio_token_id
            )
        if encoded_images is not None:
            hidden = self.inject_features(
                text_ids, hidden, self.vision_projector(encoded_images), self.config.image_token_id
            )
        bridge_states = hidden
        for index, layer in enumerate(self.thinker.layers):
            hidden = layer(hidden)
            if index == self.config.bridge_layer:
                bridge_states = hidden
        text_logits = self.thinker.lm_head(self.thinker.norm(hidden))
        audio_logits = self.talker(bridge_states, audio_code_ids, speaker_embedding, speaker_positions)
        text_loss = None
        if text_labels is not None:
            text_loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)), text_labels.reshape(-1), ignore_index=-100
            )
        audio_loss = None
        if audio_labels is not None:
            layer_losses = [
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), audio_labels[:, index].reshape(-1), ignore_index=-100
                )
                for index, logits in enumerate(audio_logits)
            ]
            audio_loss = torch.stack(layer_losses).mean()
        return OmniOutput(text_logits, audio_logits, text_loss, audio_loss)
