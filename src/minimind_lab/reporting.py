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
    _qualitative_rows(report, f"{base}generation", ("prompt", "completion"))


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
        ),
    )
    validate_language_evaluation(vlm, "language_regression")
    for index, row in enumerate(vlm["qualitative"]):
        recall = row["keyword_recall"]
        if recall is not None and (
            isinstance(recall, bool)
            or not isinstance(recall, (int, float))
            or not math.isfinite(float(recall))
            or not 0 <= float(recall) <= 1
        ):
            raise ValueError(f"VLM keyword recall must be null or in [0, 1]: qualitative[{index}]")
    _finite_metric(vlm, "visual_ablation.samples", minimum=1)
    for metric in (
        "correct_image_keyword_recall",
        "counterfactual_keyword_recall",
        "completion_change_rate",
    ):
        _finite_metric(vlm, f"visual_ablation.{metric}", minimum=0, maximum=1)
    _finite_metric(vlm, "visual_ablation.correct_minus_counterfactual_recall", minimum=-1, maximum=1)

    _finite_metric(video, "test_loss", minimum=0)
    validate_language_evaluation(video, "language_regression")
    for section in ("qivd_generation", "controlled_temporal"):
        _finite_metric(video, f"{section}.generated_samples", minimum=1)
        for metric in (
            "normalized_exact_match",
            "reversed_frame_exact_match",
            "token_f1",
            "reversed_frame_token_f1",
            "completion_change_rate_on_reversal",
        ):
            _finite_metric(video, f"{section}.{metric}", minimum=0, maximum=1)
        _finite_metric(video, f"{section}.normal_minus_reversed_token_f1", minimum=-1, maximum=1)
        _category_metrics(video, f"{section}.token_f1_by_category")
        _qualitative_rows(
            video,
            f"{section}.qualitative",
            ("question", "answer", "normal_completion", "reversed_completion"),
        )


def validate_qivd_manifest(manifest: dict, revision: str, video_count: int = 2900) -> None:
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
