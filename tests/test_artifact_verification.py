import pytest
import torch

from minimind_lab.training import file_sha256, verify_checkpoint, verify_resume_checkpoint


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


def resume_payload(step=32):
    return {
        "model": {"weight": torch.ones(2, 3)},
        "optimizer": {"state": {0: {"step": torch.tensor(1)}}, "param_groups": [{"params": [0]}]},
        "config": {"training": {"steps": 100, "gradient_accumulation_steps": 8}},
        "step": step,
        "history": [{"step": 1}, {"step": min(step, 20)}],
        "training_seconds": 12.5,
        "torch_rng_state": torch.get_rng_state(),
    }


def test_resume_verification_checks_recoverable_state(tmp_path):
    path = tmp_path / "model.resume.pt"
    payload = resume_payload()
    torch.save(payload, path)
    result = verify_resume_checkpoint(path, expected_config=payload["config"])
    assert result["resume_step"] == 32
    assert result["optimizer_boundary"] is True
    assert result["optimizer_state_entries"] == 1
    assert result["history_records"] == 2
    assert result["training_seconds"] == 12.5
    assert result["all_finite"] is True
    assert result["training_complete"] is False


def test_resume_verification_can_require_final_step(tmp_path):
    path = tmp_path / "model.resume.pt"
    payload = resume_payload(step=32)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="final step"):
        verify_resume_checkpoint(path, require_complete=True)
    payload["step"] = 100
    torch.save(payload, path)
    assert verify_resume_checkpoint(path, require_complete=True)["training_complete"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("torch_rng_state"), "missing keys"),
        (lambda value: value.update(step=31), "optimizer boundary"),
        (lambda value: value.update(training_seconds=0), "cumulative training time"),
        (lambda value: value["optimizer"].update(state={}), "optimizer state is empty"),
        (lambda value: value.update(history=[{"step": 20}, {"step": 10}]), "history steps"),
    ],
)
def test_resume_verification_rejects_incomplete_state(tmp_path, mutation, message):
    path = tmp_path / "bad.resume.pt"
    payload = resume_payload()
    mutation(payload)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        verify_resume_checkpoint(path)
