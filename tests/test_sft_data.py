import json

from minimind_lab.data import assistant_token_labels, assistant_token_ranges, normalize_conversations


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


def test_assistant_ranges_include_turn_terminator():
    tokens = [1, 9, 30, 2, 10, 1, 9, 40, 41, 2, 10]
    assert assistant_token_ranges(tokens, [1, 9], [2, 10], max_length=20) == [(2, 5), (7, 11)]


def test_tool_metadata_strings_are_normalized():
    tools = [{"type": "function", "function": {"name": "weather"}}]
    calls = [{"function": {"name": "weather", "arguments": "{}"}}]
    raw = [
        {"role": "system", "content": "", "tools": json.dumps(tools)},
        {"role": "assistant", "content": "checking", "tool_calls": json.dumps(calls)},
    ]
    messages, normalized_tools = normalize_conversations(raw)
    assert normalized_tools == tools
    assert "tools" not in messages[0]
    assert messages[1]["tool_calls"] == calls
