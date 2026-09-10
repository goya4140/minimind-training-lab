from pathlib import Path

import pytest
import torch

from minimind_lab.training import (
    ActiveTrainingTimer,
    acquire_run_lock,
    optimizer_step_size,
    rescale_partial_accumulation,
    should_save_resume,
)


def test_active_training_timer_excludes_suspend_sized_gaps():
    readings = iter([100.0, 100.5, 101.0, 33_401.0, 33_401.5])
    timer = ActiveTrainingTimer(
        prior_training_seconds=20.0,
        prior_suspended_seconds=3.0,
        maximum_step_gap_seconds=60.0,
        clock=lambda: next(readings),
    )
    assert timer.tick() == 0.5
    assert timer.tick() == 0.5
    assert timer.tick() == 0.0
    assert timer.tick() == 0.5
    assert timer.training_seconds == 21.5
    assert timer.suspended_seconds == 33_303.0


def test_run_lock_rejects_a_second_writer(tmp_path: Path):
    lock_path = tmp_path / "experiment.lock"
    first = acquire_run_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            acquire_run_lock(lock_path)
    finally:
        first.close()

    replacement = acquire_run_lock(lock_path)
    replacement.close()


def test_optimizer_step_size_flushes_the_final_partial_batch():
    assert optimizer_step_size(31, total_steps=40, accumulation=8) == 0
    assert optimizer_step_size(32, total_steps=40, accumulation=8) == 8
    assert optimizer_step_size(38, total_steps=38, accumulation=8) == 6
    with pytest.raises(ValueError, match="must be positive"):
        optimizer_step_size(1, total_steps=1, accumulation=0)


def test_partial_accumulation_is_rescaled_to_a_mean():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(0.75)
    rescale_partial_accumulation([parameter], accumulated=3, accumulation=4)
    assert parameter.grad.item() == pytest.approx(1.0)


def test_resume_checkpoint_is_only_saved_at_optimizer_boundaries():
    assert not should_save_resume(step=2000, save_interval=2000, accumulation=32)
    assert should_save_resume(step=4000, save_interval=2000, accumulation=32)
    assert should_save_resume(step=1000, save_interval=1000, accumulation=8)
