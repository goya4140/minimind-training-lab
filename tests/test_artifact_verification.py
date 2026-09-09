import torch

from minimind_lab.training import file_sha256, verify_checkpoint


def test_checkpoint_verification_hashes_and_counts_finite_state(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model": {"weight": torch.ones(2, 3), "bias": torch.zeros(2)}}, path)
    result = verify_checkpoint(path)
    assert result["checkpoint_sha256"] == file_sha256(path)
    assert result["state_tensors"] == 2
    assert result["state_parameters"] == 8
    assert result["all_finite"] is True


def test_checkpoint_verification_detects_non_finite_state(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"model": {"weight": torch.tensor([float("nan")])}}, path)
    assert verify_checkpoint(path)["all_finite"] is False


def test_checkpoint_verification_deduplicates_tied_storage(tmp_path):
    path = tmp_path / "tied.pt"
    weight = torch.ones(4, 3)
    torch.save({"model": {"embedding": weight, "head": weight}}, path)
    result = verify_checkpoint(path)
    assert result["state_tensors"] == 2
    assert result["state_parameters"] == 12
