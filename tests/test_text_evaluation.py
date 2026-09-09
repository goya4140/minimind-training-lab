import pytest

from minimind_lab.evaluation import distinct_n


def test_distinct_n_reports_unique_ngram_fraction():
    assert distinct_n([1, 2, 1, 2], n=2) == pytest.approx(2 / 3)
    assert distinct_n([1], n=2) == 0.0


def test_distinct_n_rejects_invalid_n():
    with pytest.raises(ValueError):
        distinct_n([1, 2], n=0)
