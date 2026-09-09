import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from minimind_lab.data.video import QIVDVideoDataset, qivd_split_indices, uniform_frame_indices


class TinyTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return type("Tokens", (), {"input_ids": [1, 2]})()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del messages, tokenize, add_generation_prompt
        return "prompt"


def write_tiny_qivd(root, rows=8):
    metadata = [
        {"id": index, "video_file_name": f"videos/{index}.mp4", "question": "what?", "answer": "answer"}
        for index in range(rows)
    ]
    pq.write_table(pa.Table.from_pylist(metadata), root / "metadata.parquet")


def cached_dataset(root, cache):
    return QIVDVideoDataset(
        root=root,
        tokenizer=TinyTokenizer(),
        image_processor=None,
        sequence_length=8,
        num_frames=2,
        num_video_tokens=2,
        split="train",
        train_samples=4,
        validation_samples=2,
        require_files=False,
        feature_cache=cache,
    )


def test_uniform_frame_indices_include_endpoints_and_duplicate_short_clips():
    assert uniform_frame_indices(9, 5) == [0, 2, 4, 6, 8]
    assert uniform_frame_indices(2, 4) == [0, 0, 1, 1]
    assert uniform_frame_indices(9, 1) == [4]


def test_uniform_frame_indices_reject_invalid_sizes():
    with pytest.raises(ValueError):
        uniform_frame_indices(0, 8)
    with pytest.raises(ValueError):
        uniform_frame_indices(8, 0)


def test_qivd_split_is_stable_disjoint_and_complete():
    rows = [{"id": index} for index in range(20)]
    train = qivd_split_indices(rows, "train", seed=48, train_samples=12, validation_samples=4)
    validation = qivd_split_indices(rows, "validation", seed=48, train_samples=12, validation_samples=4)
    test = qivd_split_indices(rows, "test", seed=48, train_samples=12, validation_samples=4)
    assert len(train) == 12
    assert len(validation) == 4
    assert len(test) == 4
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert sorted(train + validation + test) == list(range(20))


def test_qivd_split_rejects_unknown_split():
    rows = [{"id": index} for index in range(5)]
    with pytest.raises(ValueError, match="unknown QIVD split"):
        qivd_split_indices(rows, "other", train_samples=3, validation_samples=1)


def test_qivd_feature_cache_rejects_incomplete_bitmap(tmp_path):
    write_tiny_qivd(tmp_path)
    cache = tmp_path / "features.npy"
    np.save(cache, np.zeros((8, 2, 3, 4), dtype=np.float16))
    np.save(cache.with_suffix(".done.npy"), np.array([True] * 7 + [False]))

    with pytest.raises(RuntimeError, match="cache is incomplete"):
        cached_dataset(tmp_path, cache)


def test_qivd_feature_cache_rejects_wrong_bitmap_or_tensor_shape(tmp_path):
    write_tiny_qivd(tmp_path)
    cache = tmp_path / "features.npy"
    np.save(cache, np.zeros((8, 2, 3, 4), dtype=np.float16))
    np.save(cache.with_suffix(".done.npy"), np.ones(7, dtype=np.bool_))
    with pytest.raises(RuntimeError, match="cache is incomplete"):
        cached_dataset(tmp_path, cache)

    np.save(cache.with_suffix(".done.npy"), np.ones(8, dtype=np.bool_))
    np.save(cache, np.zeros((8, 2, 4), dtype=np.float16))
    with pytest.raises(ValueError, match="shape does not match"):
        cached_dataset(tmp_path, cache)


def test_qivd_feature_cache_loads_by_metadata_row_without_decoding(tmp_path):
    write_tiny_qivd(tmp_path)
    cache = tmp_path / "features.npy"
    features = np.arange(8 * 2 * 3 * 4, dtype=np.float16).reshape(8, 2, 3, 4)
    np.save(cache, features)
    np.save(cache.with_suffix(".done.npy"), np.ones(8, dtype=np.bool_))

    dataset = cached_dataset(tmp_path, cache)
    sample = dataset[0]
    row_index = dataset.indices[0]
    assert sample["video_inputs"].dtype == torch.float32
    assert torch.equal(sample["video_inputs"], torch.tensor(features[row_index], dtype=torch.float32))
