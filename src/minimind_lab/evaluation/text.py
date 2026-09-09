from __future__ import annotations

import re
from collections import Counter


def distinct_n(token_ids: list[int], n: int = 2) -> float:
    """Fraction of unique n-grams; zero when no complete n-gram exists."""
    if n < 1:
        raise ValueError("n must be positive")
    total = len(token_ids) - n + 1
    if total <= 0:
        return 0.0
    ngrams = {tuple(token_ids[index : index + n]) for index in range(total)}
    return len(ngrams) / total


def normalize_qa_answer(text: str) -> str:
    """Case-fold and remove punctuation for lightweight generative QA metrics."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-token F1 used by the Video-QA evaluation."""
    prediction_tokens = normalize_qa_answer(prediction).split()
    reference_tokens = normalize_qa_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
