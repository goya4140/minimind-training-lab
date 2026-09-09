from __future__ import annotations

import math
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
