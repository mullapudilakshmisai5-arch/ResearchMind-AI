"""
ResearchMind-AI
Phase 9: Research Intelligence

Provides query analysis, query expansion,
advanced retrieval, and research-oriented
answer generation.
"""

import re
from typing import List, Dict, Any


def analyze_query(query: str) -> Dict[str, Any]:
    """
    Analyze a research query and identify
    important keywords and query characteristics.
    """

    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    query = query.strip()

    if not query:
        return {
            "query": "",
            "keywords": [],
            "word_count": 0,
            "question_type": "unknown"
        }

    words = re.findall(r"\b[a-zA-Z0-9]+\b", query)

    stop_words = {
        "the", "is", "are", "was", "were",
        "a", "an", "and", "or", "of", "to",
        "in", "on", "for", "with", "what",
        "how", "why", "when", "where", "which"
    }

    keywords = [
        word.lower()
        for word in words
        if word.lower() not in stop_words
    ]

    lower_query = query.lower()

    if lower_query.startswith(("what", "define", "explain")):
        question_type = "definition"
    elif lower_query.startswith(("how", "describe")):
        question_type = "explanation"
    elif lower_query.startswith(("why", "reason")):
        question_type = "reasoning"
    elif lower_query.startswith(("compare", "difference", "differentiate")):
        question_type = "comparison"
    else:
        question_type = "general"

    return {
        "query": query,
        "keywords": list(dict.fromkeys(keywords)),
        "word_count": len(words),
        "question_type": question_type
    }


def expand_query(query: str) -> List[str]:
    """
    Generate simple research-oriented query variations.
    """

    analysis = analyze_query(query)

    original = analysis["query"]
    keywords = analysis["keywords"]

    queries = [original]

    if keywords:
        queries.append(" ".join(keywords))

    if analysis["question_type"] == "definition":
        queries.append(f"{original} definition explanation")

    elif analysis["question_type"] == "comparison":
        queries.append(f"{original} comparison differences")

    elif analysis["question_type"] == "reasoning":
        queries.append(f"{original} causes reasons")

    elif analysis["question_type"] == "explanation":
        queries.append(f"{original} detailed explanation")

    return list(dict.fromkeys(queries))


def rank_retrieved_results(
    results: List[Dict[str, Any]],
    query: str
) -> List[Dict[str, Any]]:
    """
    Rank retrieved research results using simple
    keyword-overlap scoring.
    """

    analysis = analyze_query(query)
    keywords = set(analysis["keywords"])

    ranked = []

    for result in results:
        text = str(
            result.get("text", result.get("content", ""))
        ).lower()

        matched = sum(
            1 for keyword in keywords
            if keyword in text
        )

        item = dict(result)
        item["keyword_score"] = matched

        ranked.append(item)

    ranked.sort(
        key=lambda item: item.get("keyword_score", 0),
        reverse=True
    )

    return ranked


def build_research_context(
    results: List[Dict[str, Any]],
    max_results: int = 5
) -> str:
    """
    Build a clean context from retrieved research results.
    """

    selected = results[:max_results]

    context_parts = []

    for index, result in enumerate(selected, start=1):

        text = result.get(
            "text",
            result.get("content", "")
        )

        if not text:
            continue

        context_parts.append(
            f"[Source {index}]\n{text}"
        )

    return "\n\n".join(context_parts)


def create_research_prompt(
    query: str,
    context: str
) -> str:
    """
    Create a research-oriented prompt using
    the retrieved context.
    """

    return f"""
You are a research assistant.

Answer the following research question using
only the provided context.

Research Question:
{query}

Retrieved Context:
{context}

Instructions:
1. Give a clear and accurate answer.
2. Use information from the retrieved context.
3. Do not invent unsupported facts.
4. Organize the answer logically.
5. Mention important points clearly.
"""


def research_query_info(query: str) -> Dict[str, Any]:
    """
    Return complete research intelligence information
    for a user query.
    """

    analysis = analyze_query(query)
    expanded_queries = expand_query(query)

    return {
        "analysis": analysis,
        "expanded_queries": expanded_queries
    }


if __name__ == "__main__":

    test_query = "What is Retrieval Augmented Generation?"

    result = research_query_info(test_query)

    print("=" * 60)
    print("PHASE 9 - RESEARCH INTELLIGENCE TEST")
    print("=" * 60)

    print("\nQuery:")
    print(result["analysis"]["query"])

    print("\nQuestion Type:")
    print(result["analysis"]["question_type"])

    print("\nKeywords:")
    print(result["analysis"]["keywords"])

    print("\nExpanded Queries:")
    for query in result["expanded_queries"]:
        print("-", query)

    print("\n" + "=" * 60)
    print("✅ PHASE 9 MODULE TEST COMPLETED")
    print("=" * 60)
