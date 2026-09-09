from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class JsonlPretrainDataset(Dataset):
    """Random-access JSONL dataset backed by byte offsets, not in-memory text."""

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

    def __len__(self) -> int:
        return len(self.offsets)

    def _file(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("rb")
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self._file()
        handle.seek(self.offsets[index])
        sample = json.loads(handle.readline())
        ids = self.tokenizer.encode(
            str(sample["text"]), add_special_tokens=False, truncation=True, max_length=self.sequence_length - 2
        )
        ids = [self.tokenizer.bos_token_id, *ids, self.tokenizer.eos_token_id]
        padding = self.sequence_length - len(ids)
        input_ids = torch.tensor([*ids, *([self.tokenizer.pad_token_id] * padding)], dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

