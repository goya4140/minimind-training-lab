from __future__ import annotations


def distinct_n(token_ids: list[int], n: int = 2) -> float:
    """Fraction of unique n-grams; zero when no complete n-gram exists."""
    if n < 1:
        raise ValueError("n must be positive")
    total = len(token_ids) - n + 1
    if total <= 0:
        return 0.0
    ngrams = {tuple(token_ids[index : index + n]) for index in range(total)}
    return len(ngrams) / total
