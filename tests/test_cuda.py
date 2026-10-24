import pytest
import torch

from minimind_lab.llm import MiniMindConfig, MiniMindForCausalLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_full_language_model_bfloat16_forward_backward():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 hardware required")
    model = MiniMindForCausalLM(MiniMindConfig()).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(20, 6400, (1, 340), device="cuda")
    before = model.layers[0].attention.q_proj.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(tokens, tokens)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    torch.cuda.synchronize()
    assert not torch.equal(before, model.layers[0].attention.q_proj.weight)
