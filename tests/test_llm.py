import torch

from minimind_lab.llm import ByteTokenizer, MiniMindConfig, MiniMindForCausalLM


def tiny_config() -> MiniMindConfig:
    return MiniMindConfig(
        vocab_size=259,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
    )


def test_forward_shape_and_finite_loss():
    model = MiniMindForCausalLM(tiny_config())
    tokens = torch.randint(0, 259, (2, 16))
    output = model(tokens, labels=tokens)
    assert output["logits"].shape == (2, 16, 259)
    assert torch.isfinite(output["loss"])


def test_tied_embeddings():
    model = MiniMindForCausalLM(tiny_config())
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "MiniMind 学习"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_one_optimization_step_changes_parameters():
    model = MiniMindForCausalLM(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 259, (2, 16))
    before = model.layers[0].attention.q_proj.weight.detach().clone()
    model(tokens, labels=tokens)["loss"].backward()
    optimizer.step()
    assert not torch.equal(before, model.layers[0].attention.q_proj.weight)

