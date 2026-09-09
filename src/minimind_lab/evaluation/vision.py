from __future__ import annotations


def keyword_recall(completion: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    return sum(keyword in completion for keyword in keywords) / len(keywords)


def visual_ablation_summary(cases: list[dict]) -> dict[str, float | int]:
    scored = [
        case
        for case in cases
        if case.get("keyword_recall") is not None and case.get("counterfactual_keyword_recall") is not None
    ]
    if not scored:
        raise ValueError("visual ablation has no keyword-scored cases")
    correct = sum(float(case["keyword_recall"]) for case in scored) / len(scored)
    counterfactual = sum(float(case["counterfactual_keyword_recall"]) for case in scored) / len(scored)
    return {
        "samples": len(scored),
        "correct_image_keyword_recall": correct,
        "counterfactual_keyword_recall": counterfactual,
        "correct_minus_counterfactual_recall": correct - counterfactual,
        "completion_change_rate": sum(bool(case["completion_changed_on_counterfactual"]) for case in scored)
        / len(scored),
    }
