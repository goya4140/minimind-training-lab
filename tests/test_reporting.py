from pathlib import Path

import pytest

from minimind_lab.reporting import render_temporal_ablation, render_training_curves, validate_final_evaluations


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


def valid_evaluations():
    qualitative = [
        {
            "question": "what moves?",
            "answer": "a ball",
            "normal_completion": "a ball",
            "reversed_completion": "a ball",
        }
    ]
    generation = {
        "generated_samples": 1,
        "normalized_exact_match": 1.0,
        "reversed_frame_exact_match": 0.0,
        "token_f1": 1.0,
        "reversed_frame_token_f1": 0.5,
        "normal_minus_reversed_token_f1": 0.5,
        "completion_change_rate_on_reversal": 1.0,
        "qualitative": qualitative,
    }
    llm = {
        "corpus": {"validation_loss": 2.0, "validation_perplexity": 7.4, "bits_per_byte": 1.2},
        "generation": [{"prompt": "hello", "completion": "world"}],
    }
    vlm = {
        "validation_loss": 1.5,
        "qualitative": [
            {
                "prompt": "describe",
                "completion": "a dog",
                "keyword_recall": 1.0,
                "counterfactual_completion": "a car",
                "counterfactual_keyword_recall": 0.0,
                "completion_changed_on_counterfactual": True,
            }
        ],
        "visual_ablation": {
            "samples": 1,
            "correct_image_keyword_recall": 1.0,
            "counterfactual_keyword_recall": 0.0,
            "correct_minus_counterfactual_recall": 1.0,
            "completion_change_rate": 1.0,
        },
        "language_regression": {"corpus": dict(llm["corpus"]), "generation": list(llm["generation"])},
    }
    video = {
        "test_loss": 1.0,
        "qivd_generation": generation,
        "controlled_temporal": dict(generation),
        "language_regression": {"corpus": dict(llm["corpus"]), "generation": list(llm["generation"])},
    }
    return llm, vlm, video


def test_final_evaluation_validator_accepts_complete_evidence():
    validate_final_evaluations(*valid_evaluations())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda llm, vlm, video: llm["corpus"].pop("bits_per_byte"), "missing"),
        (lambda llm, vlm, video: vlm.update(validation_loss=float("nan")), "must be finite"),
        (
            lambda llm, vlm, video: video["controlled_temporal"].update(token_f1=1.1),
            "above 1",
        ),
        (lambda llm, vlm, video: video["qivd_generation"].update(qualitative=[]), "qualitative row"),
    ],
)
def test_final_evaluation_validator_rejects_unpublishable_evidence(mutation, message):
    llm, vlm, video = valid_evaluations()
    mutation(llm, vlm, video)
    with pytest.raises(ValueError, match=message):
        validate_final_evaluations(llm, vlm, video)
