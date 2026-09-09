import pytest

from minimind_lab.training import resolve_training_steps


def test_explicit_step_budget_takes_precedence_over_epochs():
    assert resolve_training_steps({"steps": 123, "epochs": 9}, steps_per_epoch=50) == 123


def test_epoch_budget_is_supported_for_cuda_configs():
    assert resolve_training_steps({"epochs": 3}, steps_per_epoch=50) == 150


def test_training_budget_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        resolve_training_steps({"steps": 0}, steps_per_epoch=50)
