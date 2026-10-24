from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .sft import assistant_token_labels


def image_marker_count(conversations: list[dict] | str) -> int:
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    return sum(
        str(message.get("content", "")).count("<image>")
        for message in conversations
        if message.get("role") != "system"
    )


def single_image_valid_indices(path: str | Path) -> tuple[list[int], dict]:
    """Filter scalar-image Parquet rows without loading their image bytes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    image_type = parquet.schema_arrow.field("image_bytes").type
    if not (pa.types.is_binary(image_type) or pa.types.is_large_binary(image_type)):
        return list(range(parquet.metadata.num_rows)), {"policy": "multi-image-unchanged", "excluded": 0}
    valid = []
    counts = {}
    index = 0
    for batch in parquet.iter_batches(columns=["conversations"], batch_size=8192):
        for conversations in batch.column(0).to_pylist():
            count = image_marker_count(conversations)
            counts[count] = counts.get(count, 0) + 1
            if count == 1:
                valid.append(index)
            index += 1
    return valid, {
        "policy": "single-image-exactly-one-marker-v1",
        "source_rows": index,
        "retained_rows": len(valid),
        "excluded": index - len(valid),
        "marker_counts": counts,
    }


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


def truncate_vlm_tokens(input_ids: list[int], labels: list[int], *, sequence_length: int,
                        image_token_id: int, image_count: int, image_token_length: int,
                        pad_token_id: int) -> tuple[list[int], list[int]]:
    """Keep normal prefixes unchanged; repair only unsafe visual/answer crops.

    The repair retains every visual slot and one contiguous window of nonvisual
    tokens near the first answer after the last image. Labels retain their
    original positions/meaning rather than being rediscovered after cropping.
    """
    if len(input_ids) != len(labels) or sequence_length < 2 or image_count < 1:
        raise ValueError("Invalid VLM token/label lengths or image count")
    visual = [i for i, token in enumerate(input_ids) if token == image_token_id]
    expected = image_count * image_token_length
    if len(visual) != expected or image_token_length < 1:
        raise ValueError("Full VLM sequence must contain exactly all expected image slots")
    if expected >= sequence_length:
        raise ValueError("Sequence budget must leave room beyond image slots for answer supervision")
    prefix = list(range(min(sequence_length, len(input_ids))))
    safe = (sum(input_ids[i] == image_token_id for i in prefix) == expected
            and any(labels[i] != -100 and input_ids[i] != image_token_id for i in prefix[1:]))
    selected = prefix
    if not safe:
        nonvisual = [i for i, token in enumerate(input_ids) if token != image_token_id]
        budget = sequence_length - expected
        targets = [j for j, i in enumerate(nonvisual) if labels[i] != -100 and i > visual[-1]]
        if not targets:
            raise ValueError("VLM sample has no assistant target after its last image")
        start = max(0, min(targets[0] - budget // 2, len(nonvisual) - budget))
        selected = sorted(visual + nonvisual[start:start + budget])
    ids = [input_ids[i] for i in selected]
    targets = [labels[i] if input_ids[i] != image_token_id else -100 for i in selected]
    padding = sequence_length - len(ids)
    ids += [pad_token_id] * padding
    targets += [-100] * padding
    if not any(label != -100 for label in targets[1:]):
        raise ValueError("VLM crop has no next-token assistant supervision")
    return ids, targets


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
        valid_indices, self.data_quality = single_image_valid_indices(path)
        if self.data_quality["excluded"]:
            self.dataset = self.dataset.select(valid_indices)
        print(json.dumps({"vlm_data_quality": self.data_quality, "path": str(path),
                          "truncation_policy": "full-visual-slots-answer-window-v2"}), flush=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sequence_length = sequence_length
        self.image_placeholder = image_token * image_token_length
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.image_token_length = image_token_length
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
        image_bytes = row["image_bytes"]
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        full_ids = self.tokenizer(prompt).input_ids
        full_labels = assistant_token_labels(full_ids, self.assistant_start_ids, self.turn_end_ids, len(full_ids))
        input_ids, labels = truncate_vlm_tokens(
            full_ids, full_labels, sequence_length=self.sequence_length,
            image_token_id=self.image_token_id, image_count=len(image_bytes),
            image_token_length=self.image_token_length, pad_token_id=self.tokenizer.pad_token_id,
        )
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
