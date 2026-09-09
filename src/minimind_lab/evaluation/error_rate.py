from __future__ import annotations


def levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    expected = list("".join(reference.split()))
    actual = list("".join(hypothesis.split()))
    return levenshtein(expected, actual) / max(len(expected), 1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = reference.lower().split()
    actual = hypothesis.lower().split()
    return levenshtein(expected, actual) / max(len(expected), 1)
