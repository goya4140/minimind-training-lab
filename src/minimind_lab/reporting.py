from __future__ import annotations

import math
import re
from html import escape
from pathlib import Path


def _finite_points(history: list[dict], limit: int = 500) -> list[tuple[float, float]]:
    points = [
        (float(record["step"]), float(record["loss"]))
        for record in history
        if math.isfinite(float(record.get("step", math.nan))) and math.isfinite(float(record.get("loss", math.nan)))
    ]
    if len(points) <= limit:
        return points
    stride = math.ceil(len(points) / limit)
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _nested_value(report: dict, path: str):
    value = report
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"final evaluation is missing: {path}")
        value = value[key]
    return value


def _finite_metric(report: dict, path: str, minimum: float | None = None, maximum: float | None = None) -> float:
    value = _nested_value(report, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"final evaluation metric must be finite: {path}")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"final evaluation metric is below {minimum}: {path}")
    if maximum is not None and value > maximum:
        raise ValueError(f"final evaluation metric is above {maximum}: {path}")
    return value


def _qualitative_rows(report: dict, path: str, required_fields: tuple[str, ...]) -> None:
    rows = _nested_value(report, path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"final evaluation needs at least one qualitative row: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(field not in row for field in required_fields):
            raise ValueError(f"malformed qualitative row: {path}[{index}]")


def _category_metrics(report: dict, path: str) -> None:
    metrics = _nested_value(report, path)
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"final evaluation needs category metrics: {path}")
    for category, value in metrics.items():
        if not isinstance(category, str) or not category:
            raise ValueError(f"final evaluation category name is invalid: {path}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ValueError(f"final evaluation category metric must be in [0, 1]: {path}.{category}")


def validate_language_evaluation(report: dict, prefix: str = "") -> None:
    base = f"{prefix}." if prefix else ""
    _finite_metric(report, f"{base}corpus.validation_loss", minimum=0)
    _finite_metric(report, f"{base}corpus.validation_perplexity", minimum=1)
    _finite_metric(report, f"{base}corpus.bits_per_byte", minimum=0)
    generation_path = f"{base}generation"
    _qualitative_rows(
        report,
        generation_path,
        ("prompt", "completion", "new_tokens", "seconds", "tokens_per_second", "distinct_2"),
    )
    for index, row in enumerate(_nested_value(report, generation_path)):
        for field, minimum, maximum in (
            ("new_tokens", 0, None),
            ("seconds", 0, None),
            ("tokens_per_second", 0, None),
            ("distinct_2", 0, 1),
        ):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < minimum
                or (maximum is not None and float(value) > maximum)
            ):
                raise ValueError(f"language generation metric is invalid: {generation_path}[{index}].{field}")


def validate_final_evaluations(llm: dict, vlm: dict, video: dict) -> None:
    """Reject incomplete or non-finite evidence before publishing the final report."""
    validate_language_evaluation(llm)

    _finite_metric(vlm, "validation_loss", minimum=0)
    _qualitative_rows(
        vlm,
        "qualitative",
        (
            "prompt",
            "completion",
            "keyword_recall",
            "counterfactual_completion",
            "counterfactual_keyword_recall",
            "completion_changed_on_counterfactual",
            "tokens_per_second",
            "distinct_2",
        ),
    )
    validate_language_evaluation(vlm, "language_regression")
    for index, row in enumerate(vlm["qualitative"]):
        for field in ("keyword_recall", "counterfactual_keyword_recall", "distinct_2"):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"VLM metric must be in [0, 1]: qualitative[{index}].{field}")
        speed = row["tokens_per_second"]
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed < 0:
            raise ValueError(f"VLM generation speed is invalid: qualitative[{index}]")
        if not isinstance(row["completion_changed_on_counterfactual"], bool):
            raise TypeError(f"VLM counterfactual change flag is invalid: qualitative[{index}]")
    _finite_metric(vlm, "visual_ablation.samples", minimum=1)
    for metric in (
        "correct_image_keyword_recall",
        "counterfactual_keyword_recall",
        "completion_change_rate",
    ):
        _finite_metric(vlm, f"visual_ablation.{metric}", minimum=0, maximum=1)
    _finite_metric(vlm, "visual_ablation.correct_minus_counterfactual_recall", minimum=-1, maximum=1)

    _finite_metric(video, "test_loss", minimum=0)
    _finite_metric(video, "reversed_frame_test_loss", minimum=0)
    _finite_metric(video, "reversed_minus_normal_loss")
    validate_language_evaluation(video, "language_regression")
    for section in ("qivd_generation", "controlled_temporal"):
        _finite_metric(video, f"{section}.generated_samples", minimum=1)
        for metric in (
            "normalized_exact_match",
            "reversed_frame_exact_match",
            "token_f1",
            "reversed_frame_token_f1",
            "contains_reference",
            "reversed_frame_contains_reference",
            "completion_change_rate_on_reversal",
        ):
            _finite_metric(video, f"{section}.{metric}", minimum=0, maximum=1)
        _finite_metric(video, f"{section}.normal_minus_reversed_token_f1", minimum=-1, maximum=1)
        _finite_metric(video, f"{section}.mean_generation_seconds", minimum=0)
        _category_metrics(video, f"{section}.token_f1_by_category")
        _qualitative_rows(
            video,
            f"{section}.qualitative",
            (
                "category",
                "question",
                "answer",
                "normal_completion",
                "reversed_completion",
                "normal_token_f1",
                "reversed_token_f1",
            ),
        )
        for index, row in enumerate(_nested_value(video, f"{section}.qualitative")):
            for field in ("normal_token_f1", "reversed_token_f1"):
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0 <= float(value) <= 1
                ):
                    raise ValueError(f"video sample metric is invalid: {section}.qualitative[{index}].{field}")
    _finite_metric(video, "controlled_temporal.test_loss", minimum=0)
    _finite_metric(video, "controlled_temporal.reversed_frame_test_loss", minimum=0)
    _finite_metric(video, "controlled_temporal.reversed_minus_normal_loss")


def validate_evaluation_sizes(llm: dict, vlm: dict, video: dict, expected: dict[str, int]) -> None:
    """Require the fixed evaluation protocol rather than accepting a shortened run."""
    actual = {
        "llm_generation": len(_nested_value(llm, "generation")),
        "vlm_qualitative": len(_nested_value(vlm, "qualitative")),
        "qivd_test": _nested_value(video, "held_out_test_samples"),
        "qivd_generation": _nested_value(video, "qivd_generation.generated_samples"),
        "temporal_generation": _nested_value(video, "controlled_temporal.generated_samples"),
        "temporal_manifest": _nested_value(video, "controlled_temporal.manifest.samples"),
    }
    if set(actual) != set(expected):
        raise ValueError("evaluation size expectations do not match the fixed protocol")
    for name, expected_count in expected.items():
        value = actual[name]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected_count:
            raise ValueError(f"evaluation size is incorrect for {name}: expected {expected_count}, got {value}")

    if _nested_value(vlm, "visual_ablation.samples") != expected["vlm_qualitative"]:
        raise ValueError("VLM ablation sample count does not match its qualitative cases")
    for section, generated_key in (
        ("qivd_generation", "qivd_generation"),
        ("controlled_temporal", "temporal_generation"),
    ):
        qualitative = _nested_value(video, f"{section}.qualitative")
        if len(qualitative) != min(12, expected[generated_key]):
            raise ValueError(f"video qualitative sample count is incorrect for {section}")

    manifest = _nested_value(video, "controlled_temporal.manifest")
    if manifest.get("seed") != 20260909 or manifest.get("training_overlap") != 0:
        raise ValueError("controlled temporal manifest seed or training overlap is incorrect")
    expected_families = {"motion-horizontal", "motion-vertical", "size-change", "event-order"}
    if set(manifest.get("families", [])) != expected_families:
        raise ValueError("controlled temporal manifest families are incomplete")
    categories = set(_nested_value(video, "controlled_temporal.token_f1_by_category"))
    if categories != expected_families:
        raise ValueError("controlled temporal category metrics are incomplete")


def validate_prompt_alignment(*generations: list[dict]) -> None:
    """Require all language-core comparisons to use the same prompts in the same order."""
    if len(generations) < 2:
        raise ValueError("prompt alignment needs at least two generation sets")
    prompt_sets = []
    for index, rows in enumerate(generations):
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"prompt alignment generation set is empty: {index}")
        prompts = [row.get("prompt") if isinstance(row, dict) else None for row in rows]
        if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError(f"prompt alignment generation set is malformed: {index}")
        prompt_sets.append(prompts)
    if any(prompts != prompt_sets[0] for prompts in prompt_sets[1:]):
        raise ValueError("language evaluations do not use identical ordered prompts")


def validate_qivd_manifest(
    manifest: dict,
    revision: str,
    video_count: int = 2900,
    expected_root_files: dict[str, dict[str, int | str]] | None = None,
) -> None:
    """Require locally enumerated files that match the pinned upstream LFS tree."""
    if manifest.get("revision") != revision:
        raise ValueError("QIVD manifest revision does not match the pinned revision")
    if manifest.get("video_count") != video_count:
        raise ValueError("QIVD manifest video count is incorrect")
    if manifest.get("upstream_lfs_verified") is not True:
        raise ValueError("QIVD manifest lacks upstream LFS verification")
    if not isinstance(manifest.get("total_video_bytes"), int) or manifest["total_video_bytes"] <= 0:
        raise ValueError("QIVD manifest total bytes must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("aggregate_sha256", ""))):
        raise ValueError("QIVD manifest aggregate SHA-256 is malformed")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != video_count:
        raise ValueError("QIVD manifest must enumerate every video")
    if expected_root_files is not None:
        root_files = manifest.get("root_files")
        if not isinstance(root_files, list):
            raise ValueError("QIVD manifest must enumerate its root metadata and license files")
        actual = {entry.get("path"): entry for entry in root_files if isinstance(entry, dict)}
        if set(actual) != set(expected_root_files):
            raise ValueError("QIVD manifest root files are incomplete")
        for path, expected in expected_root_files.items():
            if actual[path].get("bytes") != expected["bytes"] or actual[path].get("sha256") != expected["sha256"]:
                raise ValueError(f"QIVD manifest root file does not match pinned upstream: {path}")


def validate_training_summaries(logs: dict[str, dict], expected: dict[str, tuple[int, int]]) -> None:
    """Validate completed stage summaries before condensing them into the handbook."""
    if set(logs) != set(expected):
        raise ValueError("training summary stages do not match the expected pipeline")
    for stage, (steps, trainable_parameters) in expected.items():
        report = logs[stage]
        if report.get("status") != "complete":
            raise ValueError(f"training stage is not complete: {stage}")
        if report.get("total_steps") != steps:
            raise ValueError(f"training stage step count is incorrect: {stage}")
        validation_loss = report.get("validation_loss")
        if (
            isinstance(validation_loss, bool)
            or not isinstance(validation_loss, (int, float))
            or not math.isfinite(validation_loss)
            or validation_loss < 0
        ):
            raise ValueError(f"training stage validation loss is invalid: {stage}")
        reported_parameters = report.get("trainable_parameters", report.get("parameters"))
        if reported_parameters != trainable_parameters:
            raise ValueError(f"training stage parameter count is incorrect: {stage}")
        seconds = report.get("training_seconds")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(f"training stage cumulative time is invalid: {stage}")
        suspended = report.get("suspended_seconds")
        if (
            isinstance(suspended, bool)
            or not isinstance(suspended, (int, float))
            or not math.isfinite(suspended)
            or suspended < 0
        ):
            raise ValueError(f"training stage suspended time is invalid: {stage}")
        verification = report.get("artifact_verification")
        if not isinstance(verification, dict) or verification.get("all_finite") is not True:
            raise ValueError(f"training stage checkpoint lacks finite verification: {stage}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(verification.get("checkpoint_sha256", ""))):
            raise ValueError(f"training stage checkpoint SHA-256 is malformed: {stage}")


def validate_checkpoint_bindings(logs: dict[str, dict], artifacts: dict[str, dict]) -> None:
    """Bind each completed stage report to the exact checkpoint bytes being published."""
    if set(logs) != set(artifacts):
        raise ValueError("checkpoint binding stages do not match the training summaries")
    for stage, report in logs.items():
        artifact = artifacts[stage]
        verification = report.get("artifact_verification", {})
        if report.get("checkpoint") != artifact.get("path"):
            raise ValueError(f"training report checkpoint path does not match the artifact: {stage}")
        if verification.get("checkpoint_bytes") != artifact.get("bytes"):
            raise ValueError(f"training report checkpoint size does not match the artifact: {stage}")
        if verification.get("checkpoint_sha256") != artifact.get("sha256"):
            raise ValueError(f"training report checkpoint SHA-256 does not match the artifact: {stage}")


def summarize_training_config(config: dict) -> dict[str, int | float | str]:
    """Expose the quantities a reader needs to interpret one training stage."""
    experiment = config["experiment"]
    training = config["training"]
    data = config["data"]
    micro_steps = int(training["steps"])
    micro_batch = int(training["batch_size"])
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if min(micro_steps, micro_batch, accumulation) <= 0:
        raise ValueError("training steps, batch size, and accumulation must be positive")
    return {
        "seed": int(experiment["seed"]),
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "effective_batch": micro_batch * accumulation,
        "optimizer_updates": (micro_steps + accumulation - 1) // accumulation,
        "base_learning_rate": float(training["learning_rate"]),
        "sequence_length": int(data["sequence_length"]),
        "data_path": str(data["path"]),
    }


def render_training_curves(histories: dict[str, list[dict]], output: str | Path) -> None:
    width, height = 960, 600
    columns, rows = 3, 2
    panel_width, panel_height = width / columns, height / rows
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#202124}.title{font-size:14px;font-weight:600}.axis{font-size:11px;fill:#5f6368}.line{fill:none;stroke:#2867c7;stroke-width:1.6}</style>',
    ]
    for panel_index, (name, history) in enumerate(histories.items()):
        points = _finite_points(history)
        if not points:
            raise ValueError(f"history has no finite loss points: {name}")
        column, row = panel_index % columns, panel_index // columns
        origin_x, origin_y = column * panel_width, row * panel_height
        left, right = origin_x + 48, origin_x + panel_width - 18
        top, bottom = origin_y + 38, origin_y + panel_height - 36
        x_min, x_max = points[0][0], points[-1][0]
        y_min, y_max = min(value for _, value in points), max(value for _, value in points)
        if x_max == x_min:
            x_max += 1
        if y_max == y_min:
            y_max += 1

        projected = [
            (
                left + (x - x_min) / (x_max - x_min) * (right - left),
                bottom - (y - y_min) / (y_max - y_min) * (bottom - top),
            )
            for x, y in points
        ]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in projected)
        fragments.extend(
            [
                f'<text class="title" x="{left}" y="{origin_y + 22}">{escape(name)}</text>',
                f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#dadce0"/>',
                f'<polyline class="line" points="{polyline}"/>',
                f'<text class="axis" x="{left}" y="{bottom + 18}">{int(x_min):,}</text>',
                f'<text class="axis" text-anchor="end" x="{right}" y="{bottom + 18}">{int(x_max):,} steps</text>',
                f'<text class="axis" x="{left - 5}" text-anchor="end" y="{top + 4}">{y_max:.2f}</text>',
                f'<text class="axis" x="{left - 5}" text-anchor="end" y="{bottom + 4}">{y_min:.2f}</text>',
            ]
        )
    fragments.append("</svg>")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(fragments), encoding="utf-8")


def render_temporal_ablation(metrics: dict[str, float], output: str | Path) -> None:
    labels = ["Exact", "Reversed exact", "Token F1", "Reversed F1"]
    values = [
        metrics["normalized_exact_match"],
        metrics["reversed_frame_exact_match"],
        metrics["token_f1"],
        metrics["reversed_frame_token_f1"],
    ]
    if not all(0 <= value <= 1 and math.isfinite(value) for value in values):
        raise ValueError("temporal metrics must be finite values in [0, 1]")
    width, height = 720, 360
    baseline, chart_height = 300, 230
    fragments = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#202124}.title{font-size:17px;font-weight:600}.label{font-size:12px}.value{font-size:13px;font-weight:600}</style>',
        '<text class="title" x="50" y="30">Controlled temporal benchmark: normal vs reversed frames</text>',
        f'<line x1="50" y1="{baseline}" x2="680" y2="{baseline}" stroke="#9aa0a6"/>',
    ]
    colors = ["#2867c7", "#9aa0a6", "#2867c7", "#9aa0a6"]
    for index, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        x = 85 + index * 150
        bar_height = value * chart_height
        y = baseline - bar_height
        fragments.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="86" height="{bar_height:.1f}" fill="{color}"/>',
                f'<text class="value" text-anchor="middle" x="{x + 43}" y="{max(48, y - 7):.1f}">{value:.3f}</text>',
                f'<text class="label" text-anchor="middle" x="{x + 43}" y="{baseline + 22}">{escape(label)}</text>',
            ]
        )
    fragments.append("</svg>")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(fragments), encoding="utf-8")
