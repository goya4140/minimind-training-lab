import pytest

from minimind_lab.data.video import qivd_split_indices, uniform_frame_indices


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
