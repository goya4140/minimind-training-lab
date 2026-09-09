import torch

from minimind_lab.omni import MiniMindOmni, OmniConfig


def config() -> OmniConfig:
    return OmniConfig(
        vocab_size=259,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
        talker_hidden_size=64,
        num_talker_hidden_layers=2,
        num_audio_codebooks=3,
        audio_vocab_size=80,
        audio_hidden_size=24,
        image_hidden_size=32,
        image_token_length=2,
        speaker_embedding_size=16,
        audio_speaker_token_id=79,
    )


def test_omni_forward_has_text_and_audio_losses():
    model = MiniMindOmni(config())
    text_ids = torch.randint(20, 259, (2, 12))
    text_ids[:, 1:3] = 16
    text_ids[:, 4:6] = 12
    audio_ids = torch.randint(0, 79, (2, 3, 12))
    text_labels = text_ids.clone()
    text_labels[:, :4] = -100
    audio_labels = audio_ids.clone()
    output = model(
        text_ids,
        audio_ids,
        text_labels=text_labels,
        audio_labels=audio_labels,
        encoded_audio=torch.randn(2, 2, 24),
        encoded_images=torch.randn(2, 2, 32),
    )
    assert output.text_logits.shape == (2, 12, 259)
    assert len(output.audio_logits) == 3
    assert output.audio_logits[0].shape == (2, 12, 80)
    assert torch.isfinite(output.text_loss)
    assert torch.isfinite(output.audio_loss)
    assert torch.isfinite(output.loss)


def test_talker_can_copy_thinker_tail_layers():
    model = MiniMindOmni(config())
    model.initialize_talker_from_thinker()
    source = model.thinker.layers[-2].attention.q_proj.weight
    target = model.talker.layers[0].attention.q_proj.weight
    assert torch.equal(source, target)
    assert source.data_ptr() != target.data_ptr()


def test_projector_stages_freeze_everything_else():
    model = MiniMindOmni(config())
    model.configure_trainable("audio-projector-only")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith("audio_projector.") for name in trainable)

    model.configure_trainable("vision-projector-only")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith("vision_projector.") for name in trainable)


def test_empty_audio_targets_have_zero_finite_loss():
    model = MiniMindOmni(config())
    text_ids = torch.randint(20, 259, (1, 8))
    audio_ids = torch.randint(0, 79, (1, 3, 8))
    labels = torch.full((1, 3, 8), -100)
    output = model(text_ids, audio_ids, audio_labels=labels)
    assert output.audio_loss.item() == 0.0
    assert torch.isfinite(output.audio_loss)


def test_missing_speaker_position_does_not_replace_last_token():
    model = MiniMindOmni(config())
    text_ids = torch.randint(20, 259, (1, 8))
    audio_ids = torch.randint(0, 79, (1, 3, 8))
    baseline = model(text_ids, audio_ids).audio_logits
    conditioned = model(
        text_ids,
        audio_ids,
        speaker_embedding=torch.randn(1, 16),
        speaker_positions=torch.tensor([-1]),
    ).audio_logits
    assert all(torch.equal(left, right) for left, right in zip(baseline, conditioned))
