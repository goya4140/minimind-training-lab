from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .sft import assistant_token_labels


def uniform_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    """Return deterministic, endpoint-inclusive frame indices."""
    if total_frames <= 0:
        raise ValueError("video must contain at least one frame")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return [(total_frames - 1) // 2]
    return [round(index * (total_frames - 1) / (num_frames - 1)) for index in range(num_frames)]


def decode_uniform_video(path: str | Path, num_frames: int):
    """Decode a fixed number of RGB PIL frames without retaining the full clip."""
    import av

    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total_frames = int(stream.frames or 0)
        if total_frames > 0:
            targets = uniform_frame_indices(total_frames, num_frames)
            selected = {}
            target_set = set(targets)
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in target_set:
                    selected[frame_index] = frame.to_image().convert("RGB")
                if frame_index >= targets[-1]:
                    break
            if all(index in selected for index in targets):
                return [selected[index] for index in targets]

    # Some containers do not expose a reliable frame count. Reopen and use a
    # bounded list fallback; QIVD clips are intentionally short.
    with av.open(str(path)) as container:
        frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    return [frames[index] for index in uniform_frame_indices(len(frames), num_frames)]


def qivd_split_indices(
    rows: list[dict], split: str, seed: int = 48, train_samples: int = 2400, validation_samples: int = 250
) -> list[int]:
    """Create stable, order-independent train/validation/test partitions."""
    if train_samples + validation_samples >= len(rows):
        raise ValueError("QIVD split leaves no held-out test samples")

    def split_key(index: int) -> bytes:
        row_id = rows[index].get("id", index)
        return hashlib.sha256(f"{seed}:{row_id}".encode()).digest()

    ordered = sorted(range(len(rows)), key=split_key)
    boundaries = {
        "train": ordered[:train_samples],
        "validation": ordered[train_samples : train_samples + validation_samples],
        "test": ordered[train_samples + validation_samples :],
    }
    if split not in boundaries:
        raise ValueError(f"unknown QIVD split: {split}")
    return boundaries[split]


class QIVDVideoDataset(Dataset):
    """QIVD video-question-answer examples with assistant-only text labels."""

    def __init__(
        self,
        root: str | Path,
        tokenizer,
        image_processor,
        sequence_length: int,
        num_frames: int = 8,
        num_video_tokens: int = 16,
        split: str = "train",
        split_seed: int = 48,
        train_samples: int = 2400,
        validation_samples: int = 250,
        require_files: bool = True,
        feature_cache: str | Path | None = None,
    ) -> None:
        import pyarrow.parquet as pq

        self.root = Path(root)
        self.rows = pq.read_table(self.root / "metadata.parquet").to_pylist()
        self.indices = qivd_split_indices(
            self.rows, split, split_seed, train_samples=train_samples, validation_samples=validation_samples
        )
        if require_files:
            missing = [self.rows[index]["video_file_name"] for index in self.indices if not self.video_path(index).is_file()]
            if missing:
                raise FileNotFoundError(f"QIVD split '{split}' is missing {len(missing)} videos; first: {missing[0]}")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.sequence_length = sequence_length
        self.num_frames = num_frames
        self.num_video_tokens = num_video_tokens
        self.video_placeholder = "<|video_pad|>" * num_video_tokens
        self.feature_cache = None
        if feature_cache is not None:
            import numpy as np

            feature_cache = Path(feature_cache)
            done_path = feature_cache.with_suffix(".done.npy")
            if not feature_cache.is_file() or not done_path.is_file():
                raise RuntimeError("video feature cache is incomplete; run cache_video_features.py")
            done = np.load(done_path, mmap_mode="r")
            if done.shape != (len(self.rows),) or not bool(done.all()):
                raise RuntimeError("video feature cache is incomplete; run cache_video_features.py")
            self.feature_cache = np.load(feature_cache, mmap_mode="r")
            if (
                self.feature_cache.ndim != 4
                or self.feature_cache.shape[0] != len(self.rows)
                or self.feature_cache.shape[1] != num_frames
            ):
                raise ValueError("video feature cache shape does not match QIVD metadata/config")
        self.assistant_start_ids = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.turn_end_ids = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def video_path(self, row_index: int) -> Path:
        return self.root / self.rows[row_index]["video_file_name"]

    def __len__(self) -> int:
        return len(self.indices)

    def metadata(self, index: int) -> dict:
        return self.rows[self.indices[index]]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict]:
        row_index = self.indices[index]
        row = self.rows[row_index]
        messages = [
            {"role": "user", "content": f"{self.video_placeholder}\n{row['question']}"},
            {"role": "assistant", "content": row["answer"]},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer(prompt).input_ids[: self.sequence_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.sequence_length - len(input_ids))
        labels = assistant_token_labels(input_ids, self.assistant_start_ids, self.turn_end_ids, self.sequence_length)
        if self.feature_cache is None:
            frames = decode_uniform_video(self.video_path(row_index), self.num_frames)
            video_inputs = self.image_processor(images=frames, return_tensors="pt")["pixel_values"]
        else:
            video_inputs = torch.tensor(self.feature_cache[row_index], dtype=torch.float32)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "video_inputs": video_inputs,
            "metadata": row,
        }


def collate_video(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.stack([sample["input_ids"] for sample in samples]),
        torch.stack([sample["labels"] for sample in samples]),
        torch.stack([sample["video_inputs"] for sample in samples]),
    )
