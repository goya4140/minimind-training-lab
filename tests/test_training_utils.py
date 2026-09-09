from pathlib import Path

import pytest

from minimind_lab.training import acquire_run_lock


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
