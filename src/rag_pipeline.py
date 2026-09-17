from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_llm(model_name="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


def generate_rag_answer(
    question,
    embedding_model,
    index,
    chunks,
    retrieve_context,
    tokenizer,
    model,
    top_k=5
):

    context_chunks = retrieve_context(
        question,
        embedding_model,
        index,
        chunks,
        top_k=top_k
    )

    context = "\n\n".join(
        str(chunk) for chunk in context_chunks
    )

    prompt = f"""
Use the following research paper context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer
