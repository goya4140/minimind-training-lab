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
    audio_codebook_size: int = 2048
    audio_pad_token_id: int = 2049
    audio_stop_token_id: int = 2050
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
                if position >= 0:
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
        token_ids: torch.Tensor,
        embeddings: torch.Tensor,
        features: torch.Tensor,
        placeholder_id: int,
        feature_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = []
        for batch_index in range(token_ids.size(0)):
            length = int(feature_lengths[batch_index]) if feature_lengths is not None else features.size(1)
            positions = torch.nonzero(token_ids[batch_index] == placeholder_id).flatten()
            if positions.numel() < length:
                raise ValueError("not enough modality placeholder tokens")
            item = embeddings[batch_index].clone()
            item[positions[:length]] = features[batch_index, :length]
            output.append(item)
        return torch.stack(output)

    def initialize_talker_from_thinker(self) -> None:
        if self.config.talker_hidden_size != self.config.hidden_size:
            raise ValueError("layer-copy initialization requires matching hidden sizes")
        source_start = len(self.thinker.layers) - len(self.talker.layers)
        for index, layer in enumerate(self.talker.layers):
            layer.load_state_dict(self.thinker.layers[source_start + index].state_dict())

    def configure_trainable(self, stage: str) -> None:
        """Apply the frozen-module policy used by each Omni training stage."""
        allowed = {"text-to-audio", "audio-projector-only", "vision-projector-only", "joint"}
        if stage not in allowed:
            raise ValueError(f"unknown Omni training stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad = stage in {"text-to-audio", "joint"}
        if stage == "text-to-audio":
            for module in (self.audio_projector, self.vision_projector):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        elif stage == "audio-projector-only":
            for parameter in self.audio_projector.parameters():
                parameter.requires_grad = True
        elif stage == "vision-projector-only":
            for parameter in self.vision_projector.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        text_ids: torch.Tensor,
        audio_code_ids: torch.Tensor,
        text_labels: torch.Tensor | None = None,
        audio_labels: torch.Tensor | None = None,
        encoded_audio: torch.Tensor | None = None,
        encoded_audio_lengths: torch.Tensor | None = None,
        encoded_images: torch.Tensor | None = None,
        speaker_embedding: torch.Tensor | None = None,
        speaker_positions: torch.Tensor | None = None,
    ) -> OmniOutput:
        hidden = self.thinker.embed_tokens(text_ids)
        if encoded_audio is not None:
            hidden = self.inject_features(
                text_ids,
                hidden,
                self.audio_projector(encoded_audio),
                self.config.audio_token_id,
                encoded_audio_lengths,
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
            valid_text = text_labels != -100
            if valid_text.any():
                text_loss = F.cross_entropy(text_logits[valid_text], text_labels[valid_text])
            else:
                text_loss = text_logits.sum() * 0
        audio_loss = None
        if audio_labels is not None:
            layer_losses = []
            for index, logits in enumerate(audio_logits):
                targets = audio_labels[:, index]
                valid = targets != -100
                if not valid.any():
                    layer_losses.append(logits.sum() * 0)
                    continue
                token_losses = F.cross_entropy(logits[valid], targets[valid], reduction="none")
                stop_weights = torch.where(targets[valid] == self.config.audio_stop_token_id, 10.0, 1.0)
                layer_losses.append((token_losses * stop_weights).mean())
            audio_loss = torch.stack(layer_losses).mean()
        return OmniOutput(text_logits, audio_logits, text_loss, audio_loss)

    @torch.inference_mode()
    def generate_multimodal(
        self,
        text_ids: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
        max_new_tokens: int = 256,
        text_temperature: float = 0.0,
        audio_temperature: float = 0.2,
        encoded_audio: torch.Tensor | None = None,
        encoded_audio_lengths: torch.Tensor | None = None,
        encoded_images: torch.Tensor | None = None,
        speaker_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reference autoregressive loop; correctness first, without a KV cache."""
        if text_ids.size(0) != 1:
            raise ValueError("reference Omni generation currently supports batch size 1")
        self.eval()
        prompt_length = text_ids.size(1)
        audio_ids = torch.full(
            (1, self.config.num_audio_codebooks, prompt_length),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=text_ids.device,
        )
        speaker_positions = None
        if speaker_embedding is not None and prompt_length:
            audio_ids[:, :, -1] = self.config.audio_speaker_token_id
            speaker_positions = torch.tensor([prompt_length - 1], device=text_ids.device)
        streams = [[] for _ in range(self.config.num_audio_codebooks)]
        stopped = [False] * self.config.num_audio_codebooks
        text_finished = False
        for step in range(max_new_tokens):
            output = self(
                text_ids,
                audio_ids,
                encoded_audio=encoded_audio,
                encoded_audio_lengths=encoded_audio_lengths,
                encoded_images=encoded_images,
                speaker_embedding=speaker_embedding,
                speaker_positions=speaker_positions,
            )
            text_logits = output.text_logits[:, -1]
            if text_finished:
                next_text = torch.tensor([[pad_token_id]], device=text_ids.device)
            elif text_temperature <= 0:
                next_text = text_logits.argmax(dim=-1, keepdim=True)
            else:
                next_text = torch.multinomial(F.softmax(text_logits / text_temperature, dim=-1), 1)
            if int(next_text.item()) == eos_token_id:
                text_finished = True

            next_audio = torch.full(
                (1, self.config.num_audio_codebooks, 1),
                self.config.audio_pad_token_id,
                dtype=torch.long,
                device=text_ids.device,
            )
            for codebook, logits in enumerate(output.audio_logits):
                if step <= codebook or stopped[codebook]:
                    continue
                scores = logits[:, -1]
                if audio_temperature <= 0:
                    token = scores.argmax(dim=-1, keepdim=True)
                else:
                    token = torch.multinomial(F.softmax(scores / audio_temperature, dim=-1), 1)
                value = int(token.item())
                next_audio[:, codebook, 0] = value
                if value >= self.config.audio_codebook_size:
                    stopped[codebook] = True
                else:
                    streams[codebook].append(value)
            text_ids = torch.cat((text_ids, next_text), dim=1)
            audio_ids = torch.cat((audio_ids, next_audio), dim=2)
            if text_finished and all(stopped):
                break

        generated_text = text_ids[:, prompt_length:]
        frame_count = min((len(stream) for stream in streams), default=0)
        audio_codes = torch.tensor(
            [[stream[:frame_count] for stream in streams]], dtype=torch.long, device=text_ids.device
        )
        return {"text_ids": generated_text, "audio_codes": audio_codes}
