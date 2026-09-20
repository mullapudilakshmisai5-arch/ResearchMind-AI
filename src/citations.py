def retrieve_with_citations(
    question,
    embedding_model,
    index,
    chunks,
    metadata,
    top_k=5
):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(chunks):
            continue

        item = metadata[idx] if idx < len(metadata) else {}

        if isinstance(item, dict):
            source = (
                item.get("source")
                or item.get("file_name")
                or item.get("filename")
                or "AI_RAG_paper.pdf"
            )

            page = (
                item.get("page")
                or item.get("page_number")
                or item.get("page_num")
                or item.get("page_no")
                or "Unknown"
            )
        else:
            source = "AI_RAG_paper.pdf"
            page = "Unknown"

        results.append({
            "rank": rank + 1,
            "chunk_id": int(idx),
            "text": str(chunks[idx]),
            "source": source,
            "page": page,
            "score": float(distances[0][rank])
        })

    return results


def format_citations(result):

    output = []

    output.append("Question: " + result["question"])
    output.append("")
    output.append("Answer:")
    output.append(result["answer"])
    output.append("")
    output.append("Sources:")

    for i, citation in enumerate(result["citations"], 1):

        output.append(
            f"[{i}] {citation['source']} - "
            f"Page {citation['page']}"
        )

    return "\n".join(output)
