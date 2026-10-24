import json
from pathlib import Path

import torch
import pytest

from minimind_lab.data import collate_vlm, normalize_vlm_conversations
from minimind_lab.data.vlm import single_image_valid_indices, truncate_vlm_tokens


def crop(ids, labels, length=12, images=1, slots=4):
    return truncate_vlm_tokens(ids, labels, sequence_length=length, image_token_id=12,
                               image_count=images, image_token_length=slots, pad_token_id=0)


def test_safe_visual_prefix_is_unchanged_and_padding_is_masked():
    ids = [1] + [12] * 4 + [2, 3, 4]
    labels = [-100] * 6 + [3, 4]
    assert crop(ids, labels) == (ids + [0] * 4, labels + [-100] * 4)
    ids += list(range(100, 120)); labels += list(range(100, 120))
    assert crop(ids, labels) == (ids[:12], labels[:12])


def test_long_user_prefix_does_not_cut_visual_block_or_answer():
    ids = list(range(100, 130)) + [12] * 4 + [40, 41, 42, 43]
    labels = [-100] * 34 + [40, 41, 42, 43]
    actual, targets = crop(ids, labels)
    assert len(actual) == len(targets) == 12
    assert actual.count(12) == 4
    assert targets[-4:] == [40, 41, 42, 43]
    assert [targets[i] for i, x in enumerate(actual) if x == 12] == [-100] * 4


def test_partial_visual_prefix_boundary_is_repaired():
    ids = list(range(100, 110)) + [12] * 4 + [40, 41]
    labels = [-100] * 14 + [40, 41]
    actual, targets = crop(ids, labels)
    assert actual.count(12) == 4 and targets[-2:] == [40, 41]


def test_two_images_preserve_order_and_full_slot_count():
    ids = [101, 12, 12, 102, 103, 104, 105, 106, 107, 12, 12, 108, 109]
    labels = [-100] * 11 + [108, 109]
    actual, targets = crop(ids, labels, length=10, images=2, slots=2)
    assert actual.count(12) == 4
    assert targets[-2:] == [108, 109]
    cursor = 0
    for token in actual:
        cursor = ids.index(token, cursor) + 1


def test_visual_complete_but_answer_beyond_prefix_is_repaired():
    ids = [12] * 4 + list(range(100, 140))
    labels = [-100] * 35 + ids[35:]
    actual, targets = crop(ids, labels)
    assert actual.count(12) == 4 and any(x != -100 for x in targets[1:])
    assert all(label in labels or label == -100 for label in targets)


def test_invalid_slots_budget_or_no_answer_are_rejected():
    for ids, labels, length in [([12] * 3 + [5], [-100] * 3 + [5], 12),
                                ([12] * 4 + [5], [-100] * 4 + [5], 4),
                                ([12] * 4 + [5], [-100] * 5, 12)]:
        with pytest.raises(ValueError):
            crop(ids, labels, length=length)


def test_single_image_filter_keeps_order_and_excludes_missing_or_duplicate_markers(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        [{"role": "user", "content": "<image> describe"}],
        [{"role": "user", "content": "<image><image>"}],
        [{"role": "user", "content": "no image"}],
        [{"role": "system", "content": "<image>"}, {"role": "user", "content": "<image>"}],
    ]
    path = tmp_path / "images.parquet"
    pq.write_table(pa.table({"conversations": [json.dumps(row) for row in rows], "image_bytes": [b"x"] * 4}), path)
    indices, audit = single_image_valid_indices(path)
    assert indices == [0, 3]
    assert audit["excluded"] == 2
    assert audit["marker_counts"] == {1: 2, 2: 1, 0: 1}


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
