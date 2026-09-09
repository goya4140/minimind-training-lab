from __future__ import annotations

import math
import time

import torch
from torch.utils.data import DataLoader, Subset

from minimind_lab.data import JsonlPretrainDataset

from .text import distinct_n

LANGUAGE_PROMPTS = [
    "中国的首都是",
    "请用三句话解释什么是机器学习：",
    "用户：如何制定一个可执行的学习计划？\n助手：",
    "Transformer 模型中的注意力机制",
]


@torch.inference_mode()
def language_corpus_metrics(model, loader, tokenizer, device: torch.device) -> dict[str, float | int]:
    """Evaluate token NLL, perplexity, and bits/UTF-8-byte on a held-out loader."""
    model.eval()
    negative_log_likelihood = 0.0
    predicted_tokens = 0
    utf8_bytes = 0
    started = time.perf_counter()
    special_ids = set(tokenizer.all_special_ids)
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss = model(input_ids, labels)["loss"]
        valid = labels[:, 1:] != -100
        count = int(valid.sum().item())
        negative_log_likelihood += float(loss.item()) * count
        predicted_tokens += count
        for row in labels[:, 1:].cpu().tolist():
            content_ids = [token for token in row if token != -100 and token not in special_ids]
            utf8_bytes += len(tokenizer.decode(content_ids, skip_special_tokens=True).encode("utf-8"))
    if predicted_tokens == 0 or utf8_bytes == 0:
        raise RuntimeError("language evaluation produced no predicted tokens or UTF-8 bytes")
    elapsed = time.perf_counter() - started
    mean_loss = negative_log_likelihood / predicted_tokens
    return {
        "validation_loss": mean_loss,
        "validation_perplexity": math.exp(min(mean_loss, 20)),
        "bits_per_byte": negative_log_likelihood / (math.log(2) * utf8_bytes),
        "predicted_tokens": predicted_tokens,
        "utf8_bytes": utf8_bytes,
        "validation_tokens_per_second": predicted_tokens / elapsed,
    }


@torch.inference_mode()
def language_generation_samples(
    model, tokenizer, device: torch.device, max_new_tokens: int, prompts: list[str] | None = None
) -> list[dict]:
    """Run the same deterministic text prompts for every model variant."""
    samples = []
    for prompt in prompts or LANGUAGE_PROMPTS:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        inputs = torch.tensor([[tokenizer.bos_token_id, *prompt_ids]], device=device)
        started = time.perf_counter()
        generated = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=0.0)[0]
        elapsed = time.perf_counter() - started
        completion_ids = generated[inputs.size(1) :].tolist()
        samples.append(
            {
                "prompt": prompt,
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                "new_tokens": len(completion_ids),
                "seconds": elapsed,
                "tokens_per_second": len(completion_ids) / elapsed,
                "distinct_2": distinct_n(completion_ids, n=2),
            }
        )
    return samples


def evaluate_language_regression(
    model,
    tokenizer,
    data_path,
    sequence_length: int,
    validation_samples: int,
    batch_size: int,
    device: torch.device,
    max_new_tokens: int,
) -> dict:
    """Evaluate a multimodal model's language core on the fixed LLM holdout and prompts."""
    dataset = JsonlPretrainDataset(data_path, tokenizer, sequence_length=sequence_length)
    if not 0 < validation_samples < len(dataset):
        raise ValueError("language validation samples must be positive and smaller than the dataset")
    validation = Subset(dataset, range(len(dataset) - validation_samples, len(dataset)))
    loader = DataLoader(validation, batch_size=batch_size, num_workers=0)
    return {
        "validation_samples": validation_samples,
        "corpus": language_corpus_metrics(model, loader, tokenizer, device),
        "generation": language_generation_samples(model, tokenizer, device, max_new_tokens),
    }
