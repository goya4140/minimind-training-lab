from pathlib import Path

import pytest

from minimind_lab.reporting import render_temporal_ablation, render_training_curves


def test_training_curve_renderer_writes_all_panels(tmp_path: Path):
    output = tmp_path / "curves.svg"
    render_training_curves(
        {"LLM": [{"step": 1, "loss": 4.0}, {"step": 2, "loss": 3.0}], "VLM": [{"step": 1, "loss": 2.0}]},
        output,
    )
    content = output.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "LLM" in content
    assert "VLM" in content
    assert "polyline" in content


def test_temporal_renderer_validates_metric_range(tmp_path: Path):
    output = tmp_path / "temporal.svg"
    metrics = {
        "normalized_exact_match": 0.75,
        "reversed_frame_exact_match": 0.25,
        "token_f1": 0.8,
        "reversed_frame_token_f1": 0.3,
    }
    render_temporal_ablation(metrics, output)
    assert "0.750" in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        render_temporal_ablation({**metrics, "token_f1": 1.1}, output)
