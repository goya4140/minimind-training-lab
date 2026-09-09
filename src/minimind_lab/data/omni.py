from __future__ import annotations

import io
import json

import torch
from torch.utils.data import Dataset

from .sft import assistant_token_ranges


def deinterleave_audio_codes(
    interleaved: list[int], num_codebooks: int = 8, stop_token_id: int | None = 2050
) -> list[list[int]]:
    """Convert frame-major Mimi codes into one delayed target stream per codebook."""
    usable = len(interleaved) - (len(interleaved) % num_codebooks)
    streams = [interleaved[index:usable:num_codebooks] for index in range(num_codebooks)]
    if stop_token_id is not None:
        for stream in streams:
            stream.append(stop_token_id)
    return streams


def build_delayed_audio_targets(
    code_streams: list[list[int]],
    sequence_length: int,
    assistant_start: int,
    pad_token_id: int = 2049,
    speaker_token_id: int = 2051,
    include_speaker: bool = False,
    reference_streams: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    """Build teacher-forced audio inputs and codebook-delayed labels of length T-1."""
    num_codebooks = len(code_streams)
    inputs = torch.full((num_codebooks, sequence_length), pad_token_id, dtype=torch.long)
    labels = torch.full((num_codebooks, sequence_length), -100, dtype=torch.long)

    reference_start = assistant_start
    if reference_streams:
        if len(reference_streams) != num_codebooks:
            raise ValueError("reference streams must match the number of codebooks")
        reference_length = min(len(stream) for stream in reference_streams)
        speaker_reserve = 1 if include_speaker else 0
        reference_start = max(speaker_reserve, assistant_start - reference_length)
        reference_span = assistant_start - reference_start
        for codebook, stream in enumerate(reference_streams):
            selected = stream[-reference_span:] if reference_span else []
            if selected:
                inputs[codebook, reference_start:assistant_start] = torch.tensor(selected)

    speaker_position = None
    if include_speaker and reference_start > 0:
        speaker_position = reference_start - 1
        inputs[:, speaker_position] = speaker_token_id

    for codebook, stream in enumerate(code_streams):
        start = assistant_start + codebook + 1
        available = max(sequence_length - start, 0)
        selected = stream[:available]
        if selected:
            values = torch.tensor(selected, dtype=torch.long)
            inputs[codebook, start : start + len(selected)] = values
            labels[codebook, start : start + len(selected)] = values

    return inputs[:, :-1], labels[:, 1:], speaker_position


class ParquetOmniDataset(Dataset):
    """Deterministic MiniMind-O parquet adapter without stochastic augmentation."""

    def __init__(
        self,
        path,
        tokenizer,
        sequence_length: int,
        audio_processor=None,
        vision_processor=None,
        num_codebooks: int = 8,
        audio_pad_token_id: int = 2049,
        audio_stop_token_id: int = 2050,
        audio_speaker_token_id: int = 2051,
        speaker_embedding_size: int = 192,
        image_token_length: int = 64,
    ) -> None:
        from datasets import Dataset as HFDataset

        self.dataset = HFDataset.from_parquet(str(path))
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.num_codebooks = num_codebooks
        self.audio_pad_token_id = audio_pad_token_id
        self.audio_stop_token_id = audio_stop_token_id
        self.audio_speaker_token_id = audio_speaker_token_id
        self.speaker_embedding_size = speaker_embedding_size
        self.audio_placeholder = "<|audio_pad|>"
        self.image_placeholder = "<|image_pad|>" * image_token_length
        self.assistant_start_ids = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.turn_end_ids = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.think_end_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.dataset)

    def process_audio(self, audio_bytes) -> tuple[torch.Tensor | None, int]:
        if not audio_bytes or self.audio_processor is None:
            return None, 0
        import librosa
        import numpy as np
        import soundfile as sf

        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            waveform = librosa.resample(waveform.astype(float), orig_sr=sample_rate, target_sr=16_000)
        inputs = self.audio_processor(
            waveform.astype(np.float32), sampling_rate=16_000, return_tensors="pt", return_attention_mask=True
        )
        valid_length = int(inputs.attention_mask.sum().item())
        return inputs.input_features.squeeze(0), valid_length

    def process_image(self, image_bytes) -> torch.Tensor | None:
        if not image_bytes or self.vision_processor is None:
            return None
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.vision_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        conversations = row["conversations"]
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        conversations = [dict(message) for message in conversations]
        user_count = sum(message.get("role") == "user" for message in conversations)
        assistant_count = sum(message.get("role") == "assistant" for message in conversations)

        question_audios = row.get("question_audios") or []
        question_audio = question_audios[user_count - 1] if 0 < user_count <= len(question_audios) else None
        audio_features, audio_length = self.process_audio(question_audio)
        if audio_length:
            for message in reversed(conversations):
                if message.get("role") == "user":
                    marker = self.audio_placeholder * audio_length
                    content = str(message.get("content", ""))
                    message["content"] = f"{content}\n\n{marker}" if content else marker
                    break

        image_values = row.get("image_bytes") or []
        if not isinstance(image_values, list):
            image_values = [image_values]
        pixel_values = self.process_image(image_values[0]) if image_values else None
        for message in conversations:
            if message.get("role") != "system":
                message["content"] = str(message.get("content", "")).replace("<image>", self.image_placeholder)

        prompt = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer(prompt).input_ids[: self.sequence_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.sequence_length - len(input_ids))
        ranges = assistant_token_ranges(input_ids, self.assistant_start_ids, self.turn_end_ids, self.sequence_length)
        text_labels = [-100] * self.sequence_length
        if ranges:
            start, end = ranges[-1]
            text_labels[start:end] = input_ids[start:end]
        else:
            start = self.sequence_length

        for position in range(start, min(start + 50, self.sequence_length)):
            if input_ids[position : position + len(self.think_end_ids)] == self.think_end_ids:
                start = position + len(self.think_end_ids)
                break

        answer_audios = row.get("answer_audios") or []
        answer = answer_audios[assistant_count - 1] if 0 < assistant_count <= len(answer_audios) else []
        code_streams = (
            deinterleave_audio_codes(answer, self.num_codebooks, self.audio_stop_token_id)
            if answer
            else [[] for _ in range(self.num_codebooks)]
        )
        references = row.get("ref_audios") or []
        reference_streams = (
            deinterleave_audio_codes(references, self.num_codebooks, stop_token_id=None) if references else None
        )
        speaker_raw = row.get("spk_emb") or []
        audio_ids, audio_labels, speaker_position = build_delayed_audio_targets(
            code_streams,
            self.sequence_length,
            start,
            self.audio_pad_token_id,
            self.audio_speaker_token_id,
            include_speaker=bool(speaker_raw),
            reference_streams=reference_streams,
        )
        speaker = (
            torch.tensor(speaker_raw, dtype=torch.float32) if speaker_raw else torch.zeros(self.speaker_embedding_size)
        )
        return {
            "text_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
            "text_labels": torch.tensor(text_labels[1:], dtype=torch.long),
            "audio_ids": audio_ids,
            "audio_labels": audio_labels,
            "audio_features": audio_features,
            "audio_length": audio_length,
            "pixel_values": pixel_values,
            "speaker_embedding": speaker,
            "speaker_position": speaker_position if speaker_position is not None else -1,
        }


def collate_omni(samples: list[dict]) -> dict[str, torch.Tensor | None]:
    batch = {
        "text_ids": torch.stack([sample["text_ids"] for sample in samples]),
        "text_labels": torch.stack([sample["text_labels"] for sample in samples]),
        "audio_ids": torch.stack([sample["audio_ids"] for sample in samples]),
        "audio_labels": torch.stack([sample["audio_labels"] for sample in samples]),
        "speaker_embedding": torch.stack([sample["speaker_embedding"] for sample in samples]),
        "speaker_positions": torch.tensor([sample["speaker_position"] for sample in samples]),
    }
    audio_items = [sample["audio_features"] for sample in samples]
    if any(item is not None for item in audio_items):
        template = next(item for item in audio_items if item is not None)
        max_length = max(item.size(0) if item is not None else 0 for item in audio_items)
        padded = []
        for item in audio_items:
            item = item if item is not None else template.new_zeros((0, template.size(1)))
            padded.append(torch.nn.functional.pad(item, (0, 0, 0, max_length - item.size(0))))
        batch["audio_features"] = torch.stack(padded)
        batch["audio_lengths"] = torch.tensor([sample["audio_length"] for sample in samples])
    else:
        batch["audio_features"] = batch["audio_lengths"] = None

    image_items = [sample["pixel_values"] for sample in samples]
    if any(item is not None for item in image_items):
        template = next(item for item in image_items if item is not None)
        batch["pixel_values"] = torch.stack(
            [item if item is not None else torch.zeros_like(template) for item in image_items]
        )
    else:
        batch["pixel_values"] = None
    return batch
