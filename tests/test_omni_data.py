import torch

from minimind_lab.data import build_delayed_audio_targets, deinterleave_audio_codes


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
