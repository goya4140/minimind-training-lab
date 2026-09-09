from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MiniMindConfig:
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 2432
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normalized).to(x.dtype)


def rope_frequencies(head_dim: int, length: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(length, device=device).float()
    angles = torch.outer(positions, inverse)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, heads, sequence, head_dim]
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.repeats = self.num_heads // self.num_kv_heads
        self.theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        q = self.q_proj(x).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_frequencies(self.head_dim, sequence, self.theta, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.repeats, dim=1)
        v = v.repeat_interleave(self.repeats, dim=1)
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, sequence, -1))


class SwiGLU(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.input_norm(x))
        return x + self.mlp(self.post_attention_norm(x))


class MiniMindForCausalLM(nn.Module):
    def __init__(self, config: MiniMindConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        hidden = self.dropout(self.embed_tokens(input_ids))
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64, temperature: float = 0.8) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_position_embeddings :]
            logits = self(context)["logits"][:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
