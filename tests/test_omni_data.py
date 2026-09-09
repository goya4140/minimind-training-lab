import torch

from minimind_lab.data import build_delayed_audio_targets, collate_omni, deinterleave_audio_codes


def test_mimi_codes_are_deinterleaved_and_stopped():
    streams = deinterleave_audio_codes([10, 20, 30, 11, 21, 31], num_codebooks=3, stop_token_id=79)
    assert streams == [[10, 11, 79], [20, 21, 79], [30, 31, 79]]


def test_audio_targets_follow_codebook_delay_pattern():
    streams = [[10, 11, 79], [20, 21, 79], [30, 31, 79]]
    inputs, labels, speaker_position = build_delayed_audio_targets(
        streams,
        sequence_length=12,
        assistant_start=3,
        pad_token_id=78,
        speaker_token_id=77,
        include_speaker=True,
    )
    assert speaker_position == 2
    assert torch.equal(inputs[:, 2], torch.tensor([77, 77, 77]))
    assert labels[0, 3:6].tolist() == [10, 11, 79]
    assert labels[1, 4:7].tolist() == [20, 21, 79]
    assert labels[2, 5:8].tolist() == [30, 31, 79]


def test_omni_collation_pads_optional_audio_features():
    base = {
        "text_ids": torch.ones(5, dtype=torch.long),
        "text_labels": torch.ones(5, dtype=torch.long),
        "audio_ids": torch.ones(3, 5, dtype=torch.long),
        "audio_labels": torch.ones(3, 5, dtype=torch.long),
        "pixel_values": None,
        "speaker_embedding": torch.zeros(4),
        "speaker_position": -1,
    }
    first = {**base, "audio_features": torch.ones(2, 6), "audio_length": 2}
    second = {**base, "audio_features": torch.ones(4, 6), "audio_length": 4}
    batch = collate_omni([first, second])
    assert batch["audio_features"].shape == (2, 4, 6)
    assert batch["audio_lengths"].tolist() == [2, 4]
    assert batch["pixel_values"] is None
