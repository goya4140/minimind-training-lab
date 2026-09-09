import json
from pathlib import Path

import torch

from minimind_lab.data import collate_vlm, normalize_vlm_conversations


def test_vlm_conversations_expand_images_and_tools():
    tools = [{"type": "function", "function": {"name": "locate"}}]
    conversations = [
        {"role": "system", "content": "help", "functions": json.dumps(tools)},
        {"role": "user", "content": "compare <image> with <image>"},
    ]
    messages, normalized_tools = normalize_vlm_conversations(conversations, "<pad><pad>")
    assert messages[1]["content"] == "compare <pad><pad> with <pad><pad>"
    assert normalized_tools == tools
    assert "functions" not in messages[0]


def test_vlm_collate_pads_variable_image_counts():
    sample1 = (torch.arange(4), torch.arange(4), torch.ones(1, 3, 2, 2))
    sample2 = (torch.arange(4), torch.arange(4), torch.ones(2, 3, 2, 2))
    input_ids, labels, images, counts = collate_vlm([sample1, sample2])
    assert input_ids.shape == labels.shape == (2, 4)
    assert images.shape == (2, 2, 3, 2, 2)
    assert counts.tolist() == [1, 2]
    assert images[0, 1].count_nonzero() == 0


def test_fixed_vlm_evaluation_includes_single_and_ordered_multi_image_cases():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "data/eval/vlm/prompts.json").read_text(encoding="utf-8"))
    assert sum("image" in case for case in cases) == 6
    multi_image = [case for case in cases if "images" in case]
    assert len(multi_image) == 1
    assert len(multi_image[0]["images"]) == 2
    assert all((root / "data/eval/vlm" / name).is_file() for name in multi_image[0]["images"])
