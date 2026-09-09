from __future__ import annotations

import torch


def deinterleave_audio_codes(
    interleaved: list[int], num_codebooks: int = 8, stop_token_id: int = 2050
) -> list[list[int]]:
    """Convert frame-major Mimi codes into one delayed target stream per codebook."""
    usable = len(interleaved) - (len(interleaved) % num_codebooks)
    streams = [interleaved[index:usable:num_codebooks] for index in range(num_codebooks)]
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
