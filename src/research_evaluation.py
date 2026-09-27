import json
import os
from typing import List, Dict, Any


def evaluate_research_output(
    answer: str,
    retrieved_context: List[Dict[str, Any]],
    citations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate a generated research answer using simple
    quality indicators.
    """

    answer = answer.strip() if answer else ""

    context_count = len(retrieved_context)
    citation_count = len(citations)

    answer_length = len(answer)

    if answer_length == 0:
        completeness = 0.0
    elif answer_length < 100:
        completeness = 0.4
    elif answer_length < 300:
        completeness = 0.7
    else:
        completeness = 1.0

    if context_count == 0:
        context_score = 0.0
    elif context_count < 3:
        context_score = 0.5
    else:
        context_score = 1.0

    if citation_count == 0:
        citation_score = 0.0
    elif citation_count < 2:
        citation_score = 0.5
    else:
        citation_score = 1.0

    overall_score = round(
        (completeness + context_score + citation_score) / 3,
        2
    )

    return {
        "answer_length": answer_length,
        "retrieved_context_count": context_count,
        "citation_count": citation_count,
        "completeness_score": completeness,
        "context_score": context_score,
        "citation_score": citation_score,
        "overall_score": overall_score
    }


def save_evaluation(
    evaluation: Dict[str, Any],
    output_path: str = "data/processed/evaluation.json"
) -> str:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    return output_path
