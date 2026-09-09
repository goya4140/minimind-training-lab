from pathlib import Path

import pytest

from minimind_lab.reporting import (
    render_temporal_ablation,
    render_training_curves,
    summarize_training_config,
    validate_checkpoint_bindings,
    validate_evaluation_sizes,
    validate_final_evaluations,
    validate_prompt_alignment,
    validate_qivd_manifest,
    validate_training_summaries,
)


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
            "category": "motion",
            "question": "what moves?",
            "answer": "a ball",
            "normal_completion": "a ball",
            "reversed_completion": "a ball",
            "normal_token_f1": 1.0,
            "reversed_token_f1": 0.5,
        }
    ]
    generation = {
        "generated_samples": 1,
        "normalized_exact_match": 1.0,
        "reversed_frame_exact_match": 0.0,
        "contains_reference": 1.0,
        "reversed_frame_contains_reference": 0.0,
        "token_f1": 1.0,
        "reversed_frame_token_f1": 0.5,
        "normal_minus_reversed_token_f1": 0.5,
        "completion_change_rate_on_reversal": 1.0,
        "mean_generation_seconds": 0.5,
        "token_f1_by_category": {"motion": 1.0},
        "qualitative": qualitative,
    }
    llm = {
        "corpus": {"validation_loss": 2.0, "validation_perplexity": 7.4, "bits_per_byte": 1.2},
        "generation": [
            {
                "prompt": "hello",
                "completion": "world",
                "new_tokens": 1,
                "seconds": 0.1,
                "tokens_per_second": 10.0,
                "distinct_2": 0.0,
            }
        ],
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
                "tokens_per_second": 10.0,
                "distinct_2": 0.5,
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
        "reversed_frame_test_loss": 1.1,
        "reversed_minus_normal_loss": 0.1,
        "qivd_generation": generation,
        "controlled_temporal": {
            **generation,
            "test_loss": 1.0,
            "reversed_frame_test_loss": 1.2,
            "reversed_minus_normal_loss": 0.2,
        },
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
        (
            lambda llm, vlm, video: video["qivd_generation"].update(token_f1_by_category={}),
            "category metrics",
        ),
        (
            lambda llm, vlm, video: llm["generation"][0].update(tokens_per_second=float("nan")),
            "language generation metric",
        ),
        (
            lambda llm, vlm, video: video["controlled_temporal"]["qualitative"][0].update(
                normal_token_f1=1.1
            ),
            "video sample metric",
        ),
    ],
)
def test_final_evaluation_validator_rejects_unpublishable_evidence(mutation, message):
    llm, vlm, video = valid_evaluations()
    mutation(llm, vlm, video)
    with pytest.raises(ValueError, match=message):
        validate_final_evaluations(llm, vlm, video)


def test_vlm_counterfactual_change_flag_must_be_boolean():
    llm, vlm, video = valid_evaluations()
    vlm["qualitative"][0]["completion_changed_on_counterfactual"] = "yes"
    with pytest.raises(TypeError, match="change flag"):
        validate_final_evaluations(llm, vlm, video)


def test_qivd_manifest_validator_requires_pinned_upstream_evidence():
    revision = "a" * 40
    manifest = {
        "revision": revision,
        "video_count": 2,
        "total_video_bytes": 10,
        "aggregate_sha256": "b" * 64,
        "upstream_lfs_verified": True,
        "files": [{"path": "0.mp4"}, {"path": "1.mp4"}],
    }
    validate_qivd_manifest(manifest, revision, video_count=2)
    with pytest.raises(ValueError, match="upstream LFS"):
        validate_qivd_manifest({**manifest, "upstream_lfs_verified": False}, revision, video_count=2)
    with pytest.raises(ValueError, match="enumerate every video"):
        validate_qivd_manifest({**manifest, "files": []}, revision, video_count=2)


def test_training_summary_validator_locks_steps_parameters_time_and_checkpoint():
    report = {
        "status": "complete",
        "total_steps": 100,
        "parameters": 200,
        "validation_loss": 1.0,
        "training_seconds": 30.0,
        "artifact_verification": {"all_finite": True, "checkpoint_sha256": "a" * 64},
    }
    validate_training_summaries({"llm": report}, {"llm": (100, 200)})
    with pytest.raises(ValueError, match="step count"):
        validate_training_summaries({"llm": {**report, "total_steps": 99}}, {"llm": (100, 200)})
    with pytest.raises(ValueError, match="cumulative time"):
        validate_training_summaries({"llm": {**report, "training_seconds": 0}}, {"llm": (100, 200)})
    with pytest.raises(ValueError, match="validation loss"):
        validate_training_summaries({"llm": {**report, "validation_loss": float("nan")}}, {"llm": (100, 200)})
    with pytest.raises(ValueError, match="finite verification"):
        validate_training_summaries(
            {"llm": {**report, "artifact_verification": {"all_finite": False}}}, {"llm": (100, 200)}
        )


def test_checkpoint_binding_requires_exact_path_size_and_hash():
    report = {
        "checkpoint": "artifacts/checkpoints/model.pt",
        "artifact_verification": {
            "checkpoint_bytes": 123,
            "checkpoint_sha256": "a" * 64,
        },
    }
    artifact = {
        "path": "artifacts/checkpoints/model.pt",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    validate_checkpoint_bindings({"stage": report}, {"stage": artifact})
    with pytest.raises(ValueError, match="SHA-256"):
        validate_checkpoint_bindings(
            {"stage": report},
            {"stage": {**artifact, "sha256": "b" * 64}},
        )
    with pytest.raises(ValueError, match="size"):
        validate_checkpoint_bindings(
            {"stage": report},
            {"stage": {**artifact, "bytes": 124}},
        )
    with pytest.raises(ValueError, match="path"):
        validate_checkpoint_bindings(
            {"stage": report},
            {"stage": {**artifact, "path": "artifacts/checkpoints/other.pt"}},
        )


def test_training_config_summary_exposes_effective_batch_and_updates():
    config = {
        "experiment": {"seed": 42},
        "training": {"steps": 101, "batch_size": 8, "gradient_accumulation_steps": 32, "learning_rate": 5e-4},
        "data": {"sequence_length": 340, "path": "data/train.jsonl"},
    }
    assert summarize_training_config(config) == {
        "seed": 42,
        "micro_batch": 8,
        "accumulation": 32,
        "effective_batch": 256,
        "optimizer_updates": 4,
        "base_learning_rate": 5e-4,
        "sequence_length": 340,
        "data_path": "data/train.jsonl",
    }


def test_evaluation_size_validator_rejects_shortened_protocol():
    llm, vlm, video = valid_evaluations()
    video["held_out_test_samples"] = 250
    video["controlled_temporal"]["manifest"] = {
        "samples": 1,
        "seed": 20260909,
        "training_overlap": 0,
        "families": ["motion-horizontal", "motion-vertical", "size-change", "event-order"],
    }
    video["controlled_temporal"]["token_f1_by_category"] = {
        "motion-horizontal": 1.0,
        "motion-vertical": 1.0,
        "size-change": 1.0,
        "event-order": 1.0,
    }
    expected = {
        "llm_generation": 1,
        "vlm_qualitative": 1,
        "qivd_test": 250,
        "qivd_generation": 1,
        "temporal_generation": 1,
        "temporal_manifest": 1,
    }
    validate_evaluation_sizes(llm, vlm, video, expected)
    with pytest.raises(ValueError, match="qivd_generation"):
        validate_evaluation_sizes(llm, vlm, video, {**expected, "qivd_generation": 100})


def test_language_prompt_alignment_requires_identical_order():
    first = [{"prompt": "a"}, {"prompt": "b"}]
    validate_prompt_alignment(first, [{"prompt": "a"}, {"prompt": "b"}])
    with pytest.raises(ValueError, match="identical ordered prompts"):
        validate_prompt_alignment(first, [{"prompt": "b"}, {"prompt": "a"}])
