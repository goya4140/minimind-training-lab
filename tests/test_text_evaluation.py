import pytest
import torch

from minimind_lab.evaluation import (
    distinct_n,
    language_corpus_metrics,
    language_generation_samples,
    normalize_qa_answer,
    token_f1,
)


class TinyLanguageTokenizer:
    all_special_ids = (0, 1)
    bos_token_id = 1

    def encode(self, prompt, add_special_tokens=False):
        del prompt, add_special_tokens
        return [2]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(str(token) for token in token_ids)


class TinyLanguageModel:
    def eval(self):
        return self

    def __call__(self, input_ids, labels):
        del input_ids, labels
        return {"loss": torch.tensor(1.0)}

    def generate(self, inputs, max_new_tokens, temperature):
        del max_new_tokens, temperature
        return torch.cat((inputs, torch.tensor([[3, 4]], device=inputs.device)), dim=1)


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


def test_shared_language_metrics_and_generation_are_model_agnostic():
    tokenizer = TinyLanguageTokenizer()
    model = TinyLanguageModel()
    loader = [(torch.tensor([[1, 2, 3, 0]]), torch.tensor([[1, 2, 3, -100]]))]

    metrics = language_corpus_metrics(model, loader, tokenizer, torch.device("cpu"))
    samples = language_generation_samples(
        model, tokenizer, torch.device("cpu"), max_new_tokens=2, prompts=["same prompt"]
    )

    assert metrics["validation_loss"] == pytest.approx(1.0)
    assert metrics["validation_perplexity"] == pytest.approx(2.718281828)
    assert metrics["predicted_tokens"] == 2
    assert samples[0]["prompt"] == "same prompt"
    assert samples[0]["completion"] == "34"
