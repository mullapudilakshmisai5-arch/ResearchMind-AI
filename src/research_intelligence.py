
import json
import os


def analyze_query(question):
    question_lower = question.lower().strip()

    if any(word in question_lower for word in [
        "what is", "define", "definition", "meaning"
    ]):
        query_type = "definition"

    elif any(word in question_lower for word in [
        "how does", "how do", "explain", "working", "works"
    ]):
        query_type = "explanation"

    elif any(word in question_lower for word in [
        "benefit", "advantages", "advantage",
        "useful", "importance"
    ]):
        query_type = "benefits"

    elif any(word in question_lower for word in [
        "difference", "compare", "comparison",
        "versus", "vs"
    ]):
        query_type = "comparison"

    elif any(word in question_lower for word in [
        "limitation", "limitations",
        "disadvantage", "disadvantages", "problem"
    ]):
        query_type = "limitations"

    else:
        query_type = "general"

    return {
        "question": question,
        "query_type": query_type,
        "keywords": question_lower.split()
    }


def expand_query(question):
    question_lower = question.lower().strip()

    expanded_terms = [question]

    if "retrieval" in question_lower or "rag" in question_lower:
        expanded_terms.extend([
            "retrieval augmented generation",
            "retrieval augmented generation RAG",
            "retrieval based language generation",
            "retrieving external knowledge for generation"
        ])

    if (
        "transformer" in question_lower
        or "language model" in question_lower
        or "llm" in question_lower
    ):
        expanded_terms.extend([
            "transformer language model",
            "large language model",
            "neural language model"
        ])

    return list(dict.fromkeys(expanded_terms))


def advanced_retrieval(
    question,
    embedding_model,
    index,
    chunks,
    top_k=5
):
    queries = expand_query(question)

    all_results = []

    for query in queries:

        query_embedding = embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = index.search(
            query_embedding,
            top_k
        )

        for rank, idx in enumerate(indices[0]):

            if idx < 0 or idx >= len(chunks):
                continue

            all_results.append({
                "chunk_id": int(idx),
                "text": str(chunks[idx]),
                "score": float(distances[0][rank])
            })

    unique_results = {}

    for result in all_results:

        chunk_id = result["chunk_id"]

        if chunk_id not in unique_results:
            unique_results[chunk_id] = result

    results = list(unique_results.values())

    return results[:top_k]


def advanced_rag_with_citations(
    question,
    embedding_model,
    index,
    chunks,
    citation_metadata,
    tokenizer,
    model,
    top_k=5
):

    results = advanced_retrieval(
        question,
        embedding_model,
        index,
        chunks,
        top_k
    )

    cited_results = []

    for result in results:

        chunk_id = result["chunk_id"]

        citation_info = citation_metadata.get(
            chunk_id,
            {}
        )

        cited_results.append({
            "chunk_id": chunk_id,
            "text": result["text"],
            "score": result["score"],
            "source": citation_info.get(
                "source",
                "Unknown"
            ),
            "page": citation_info.get(
                "page",
                "Unknown"
            )
        })

    context = "\n\n".join(
        result["text"]
        for result in cited_results
    )

    query_info = analyze_query(question)

    query_type = query_info["query_type"]

    prompt = f"""
You are an AI research assistant.

Question type:
{query_type}

Use only the research context provided below.

Research Context:
{context}

Question:
{question}

Give a clear and accurate answer.
Do not invent information.

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=250
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    citations = []

    for result in cited_results:

        citations.append({
            "chunk_id": result["chunk_id"],
            "source": result["source"],
            "page": result["page"],
            "score": result["score"]
        })

    return {
        "question": question,
        "query_type": query_type,
        "answer": answer,
        "citations": citations
    }
