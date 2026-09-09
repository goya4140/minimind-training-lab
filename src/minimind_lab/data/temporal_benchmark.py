from __future__ import annotations

import json
import os
import random
from pathlib import Path

BENCHMARK_VERSION = 2


def _write_video(path: Path, images: list, fps: int = 4) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    with av.open(str(temporary), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = images[0].width
        stream.height = images[0].height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for image in images:
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    os.replace(temporary, path)


def _canvas(color: tuple[int, int, int]):
    from PIL import Image

    return Image.new("RGB", (256, 256), color)


def _moving_shape(rng: random.Random, direction: str) -> list:
    from PIL import ImageDraw

    background = rng.choice([(232, 238, 247), (245, 236, 222), (224, 242, 230)])
    foreground = rng.choice([(220, 50, 47), (38, 92, 210), (230, 145, 28), (126, 62, 176)])
    start, end = {
        "right": ((38, 128), (218, 128)),
        "left": ((218, 128), (38, 128)),
        "down": ((128, 38), (128, 218)),
        "up": ((128, 218), (128, 38)),
    }[direction]
    images = []
    for index in range(8):
        fraction = index / 7
        x = round(start[0] + fraction * (end[0] - start[0]))
        y = round(start[1] + fraction * (end[1] - start[1]))
        image = _canvas(background)
        draw = ImageDraw.Draw(image)
        radius = 18
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=foreground)
        images.append(image)
    return images


def _changing_size(rng: random.Random, change: str) -> list:
    from PIL import ImageDraw

    background = rng.choice([(239, 239, 228), (225, 236, 246), (241, 226, 236)])
    foreground = rng.choice([(30, 125, 80), (190, 55, 70), (48, 82, 190)])
    radii = [round(12 + index * 48 / 7) for index in range(8)]
    if change == "smaller":
        radii.reverse()
    images = []
    for radius in radii:
        image = _canvas(background)
        draw = ImageDraw.Draw(image)
        draw.ellipse((128 - radius, 128 - radius, 128 + radius, 128 + radius), fill=foreground)
        images.append(image)
    return images


def _event_order(rng: random.Random, first: str) -> list:
    from PIL import ImageDraw

    background = rng.choice([(236, 236, 236), (245, 239, 220), (222, 238, 242)])
    images = []
    order = [first] * 4 + [("blue circle" if first == "red square" else "red square")] * 4
    for event in order:
        image = _canvas(background)
        draw = ImageDraw.Draw(image)
        if event == "red square":
            draw.rectangle((88, 88, 168, 168), fill=(215, 45, 55))
        else:
            draw.ellipse((88, 88, 168, 168), fill=(35, 85, 215))
        images.append(image)
    return images


def generate_temporal_benchmark(root: str | Path, samples_per_family: int = 40, seed: int = 20260909) -> dict:
    """Generate a deterministic, text-free temporal video QA benchmark."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    manifest_path = root / "manifest.json"
    metadata_path = root / "metadata.parquet"
    expected = samples_per_family * 4
    if manifest_path.is_file() and metadata_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("version") == BENCHMARK_VERSION
            and manifest.get("seed") == seed
            and manifest.get("samples") == expected
            and all((root / f"videos/{index:06d}.mp4").is_file() for index in range(expected))
        ):
            return manifest

    rng = random.Random(seed)
    rows = []
    specifications = []
    for index in range(samples_per_family):
        specifications.append(("motion-horizontal", rng.choice(["left", "right"])))
        specifications.append(("motion-vertical", rng.choice(["up", "down"])))
        specifications.append(("size-change", rng.choice(["larger", "smaller"])))
        specifications.append(("event-order", rng.choice(["red square", "blue circle"])))

    for row_id, (family, answer) in enumerate(specifications):
        if family.startswith("motion"):
            frames = _moving_shape(rng, answer)
            question = "In which direction does the square move?"
            full_answer = f"The square moves {answer}."
        elif family == "size-change":
            frames = _changing_size(rng, answer)
            question = "Does the circle become larger or smaller over time?"
            full_answer = f"The circle becomes {answer}."
        else:
            frames = _event_order(rng, answer)
            question = "Which appeared first, the red square or the blue circle?"
            full_answer = f"The {answer} appeared first."
        relative_path = f"videos/{row_id:06d}.mp4"
        _write_video(root / relative_path, frames)
        rows.append(
            {
                "video_file_name": relative_path,
                "id": row_id,
                "category": family,
                "question": question,
                "answer": full_answer,
                "short_answer": answer,
                "timestamp": "00:00.0",
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    temporary_metadata = metadata_path.with_suffix(".tmp.parquet")
    pq.write_table(pa.Table.from_pylist(rows), temporary_metadata)
    os.replace(temporary_metadata, metadata_path)
    manifest = {
        "name": "MiniMind Controlled Temporal Video QA",
        "version": BENCHMARK_VERSION,
        "seed": seed,
        "samples_per_family": samples_per_family,
        "samples": len(rows),
        "families": ["motion-horizontal", "motion-vertical", "size-change", "event-order"],
        "training_overlap": 0,
    }
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest
