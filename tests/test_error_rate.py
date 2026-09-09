from minimind_lab.evaluation import character_error_rate, levenshtein, word_error_rate


def test_levenshtein_counts_insertions_deletions_and_substitutions():
    assert levenshtein(list("kitten"), list("sitting")) == 3


def test_character_error_rate_ignores_whitespace():
    assert character_error_rate("你 好", "你号") == 0.5


def test_word_error_rate_is_case_insensitive():
    assert word_error_rate("Hello small world", "hello world") == 1 / 3
