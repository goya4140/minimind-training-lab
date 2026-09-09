from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .sft import assistant_token_labels


def normalize_vlm_conversations(
    conversations: list[dict], image_placeholder: str
) -> tuple[list[dict], list[dict] | None]:
    messages = []
    tools = None
    for raw_message in conversations:
        message = dict(raw_message)
        if message.get("role") == "system" and message.get("functions"):
            value = message.pop("functions")
            tools = json.loads(value) if isinstance(value, str) else value
        if message.get("role") != "system":
            message["content"] = str(message.get("content", "")).replace("<image>", image_placeholder)
        messages.append(message)
    return messages, tools


class ParquetVLMDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        image_processor,
        sequence_length: int,
        image_token: str = "<|image_pad|>",
        image_token_length: int = 64,
    ) -> None:
        from datasets import Dataset as HFDataset

        self.dataset = HFDataset.from_parquet(str(path))
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sequence_length = sequence_length
        self.image_placeholder = image_token * image_token_length
        self.assistant_start_ids = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.turn_end_ids = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from PIL import Image

        row = self.dataset[index]
        conversations = row["conversations"]
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        messages, tools = normalize_vlm_conversations(conversations, self.image_placeholder)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        input_ids = self.tokenizer(prompt).input_ids[: self.sequence_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.sequence_length - len(input_ids))
        labels = assistant_token_labels(input_ids, self.assistant_start_ids, self.turn_end_ids, self.sequence_length)

        image_bytes = row["image_bytes"]
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        images = [Image.open(io.BytesIO(value)).convert("RGB") for value in image_bytes]
        pixel_values = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            pixel_values,
        )


def collate_vlm(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, labels, images = zip(*samples)
    counts = torch.tensor([item.size(0) for item in images], dtype=torch.long)
    max_images = int(counts.max())
    padded = []
    for item in images:
        if item.size(0) < max_images:
            padding = item.new_zeros((max_images - item.size(0), *item.shape[1:]))
            item = torch.cat((item, padding), dim=0)
        padded.append(item)
    return torch.stack(input_ids), torch.stack(labels), torch.stack(padded), counts
