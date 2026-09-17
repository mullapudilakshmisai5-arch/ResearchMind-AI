
import numpy as np


def semantic_search(query, embedding_model, index, chunks, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, idx in enumerate(indices[0], start=1):

        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        else:
            text = str(chunk)

        results.append({
            "rank": rank,
            "chunk_index": int(idx),
            "score": float(distances[0][rank - 1]),
            "text": text
        })

    return results


def retrieve_context(
    query,
    embedding_model,
    index,
    chunks,
    top_k=5
):

    results = semantic_search(
        query,
        embedding_model,
        index,
        chunks,
        top_k
    )

    context_parts = []

    for result in results:

        text = result["text"].strip()

        if text:
            context_parts.append(
                f"[Chunk {result['rank']}]\n{text}"
            )

    context = "\n\n".join(context_parts)

    return {
        "query": query,
        "results": results,
        "context": context
    }
