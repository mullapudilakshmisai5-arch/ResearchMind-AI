
# ============================================================
# RESEARCHMIND-AI - PHASE 5
# RAG RETRIEVAL PIPELINE
# ============================================================

import os
import json
import faiss
from sentence_transformers import SentenceTransformer

PROJECT_DIR = "/content/drive/MyDrive/ResearchMind-AI"

INDEX_PATH = os.path.join(
    PROJECT_DIR,
    "data/vector_db/researchmind.index"
)

METADATA_PATH = os.path.join(
    PROJECT_DIR,
    "data/vector_db/metadata.json"
)

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_relevant_chunks(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx]

        retrieved_chunks.append({
            "rank": rank,
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "document": chunk["document"],
            "page": chunk["page"],
            "text": chunk["text"]
        })

    return retrieved_chunks


def build_context(retrieved_chunks):

    context_parts = []

    for result in retrieved_chunks:

        context_part = (
            f"[Source: {result['document']}, "
            f"Page: {result['page']}]\n"
            f"{result['text']}"
        )

        context_parts.append(context_part)

    return "\n\n".join(context_parts)


def build_rag_prompt(query, context):

    prompt = f"""
You are an AI research assistant.

Answer the user's question using only the information provided
in the context below.

If the answer is not available in the context, clearly say:
"I could not find the answer in the provided research document."

Do not invent or assume information.

Always provide a clear and concise answer.

USER QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""

    return prompt.strip()
