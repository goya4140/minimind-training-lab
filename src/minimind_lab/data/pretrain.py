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


class DeterministicBatchStream:
    """Epoch-wise deterministic shuffle that can resume from any global step."""

    def __init__(self, dataset: Dataset, batch_size: int, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size
        self._epoch = -1
        self._permutation = torch.empty(0, dtype=torch.long)

    def indices_for_step(self, step: int) -> list[int]:
        if step < 1:
            raise ValueError("step is one-based and must be positive")
        epoch = (step - 1) // self.steps_per_epoch
        position = (step - 1) % self.steps_per_epoch
        if epoch != self._epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._permutation = torch.randperm(len(self.dataset), generator=generator)
            self._epoch = epoch
        start = position * self.batch_size
        return self._permutation[start : start + self.batch_size].tolist()

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        samples = [self.dataset[index] for index in self.indices_for_step(step)]
        input_ids, labels = zip(*samples)
        return torch.stack(input_ids), torch.stack(labels)
