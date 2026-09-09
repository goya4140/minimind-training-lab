import pytest

from minimind_lab.evaluation import distinct_n, normalize_qa_answer, token_f1


def test_distinct_n_reports_unique_ngram_fraction():
    assert distinct_n([1, 2, 1, 2], n=2) == pytest.approx(2 / 3)
    assert distinct_n([1], n=2) == 0.0


def test_distinct_n_rejects_invalid_n():
    with pytest.raises(ValueError):
        distinct_n([1, 2], n=0)


def test_video_qa_normalization_and_token_f1():
    assert normalize_qa_answer("The BLUE-circle!") == "the blue circle"
    assert token_f1("a blue circle", "blue circle") == pytest.approx(0.8)
    assert token_f1("red", "blue") == 0.0
    assert token_f1("", "") == 1.0
