from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def normalize_conversations(conversations: list[dict]) -> tuple[list[dict], list[dict] | None]:
    messages = []
    tools = None
    for raw_message in conversations:
        message = dict(raw_message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            message.pop("tools", None)
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return messages, tools


def assistant_token_labels(
    input_ids: list[int], assistant_start_ids: list[int], turn_end_ids: list[int], max_length: int
) -> list[int]:
    """Mask every token except assistant responses, including each response terminator."""
    labels = [-100] * len(input_ids)
    position = 0
    while position < len(input_ids):
        if input_ids[position : position + len(assistant_start_ids)] == assistant_start_ids:
            start = position + len(assistant_start_ids)
            end = start
            while end < len(input_ids):
                if input_ids[end : end + len(turn_end_ids)] == turn_end_ids:
                    break
                end += 1
            target_end = min(end + len(turn_end_ids), max_length)
            labels[start:target_end] = input_ids[start:target_end]
            position = target_end
        else:
            position += 1
    return labels


class JsonlSFTDataset(Dataset):
    """Chat-template SFT dataset with assistant-only labels."""

    def __init__(self, path: str | Path, tokenizer, sequence_length: int) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.offsets: list[int] = []
        self._handle = None
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        self.assistant_start_ids = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.turn_end_ids = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.offsets)

    def _file(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self._file()
        handle.seek(self.offsets[index])
        conversations = json.loads(handle.readline())["conversations"]
        messages, tools = normalize_conversations(conversations)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        input_ids = self.tokenizer(prompt).input_ids[: self.sequence_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.sequence_length - len(input_ids))
        labels = assistant_token_labels(input_ids, self.assistant_start_ids, self.turn_end_ids, self.sequence_length)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state
