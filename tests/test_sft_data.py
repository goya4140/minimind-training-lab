from minimind_lab.data import assistant_token_labels


def test_assistant_only_labels_cover_multiple_turns():
    start = [1, 9]
    end = [2, 10]
    tokens = [1, 7, 20, 2, 10, 1, 9, 30, 31, 2, 10, 1, 7, 21, 2, 10, 1, 9, 40, 2, 10]
    labels = assistant_token_labels(tokens, start, end, max_length=len(tokens))
    expected_targets = {7, 8, 9, 10, 18, 19, 20}
    assert {index for index, label in enumerate(labels) if label != -100} == expected_targets
    assert [labels[index] for index in sorted(expected_targets)] == [30, 31, 2, 10, 40, 2, 10]


def test_truncated_assistant_response_is_still_supervised():
    tokens = [1, 9, 30, 31, 32]
    labels = assistant_token_labels(tokens, [1, 9], [2, 10], max_length=5)
    assert labels == [-100, -100, 30, 31, 32]

