
import os
import json
import torch
import gradio as gr
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_PATH = os.path.join(BASE_DIR, "data/processed/chunks.json")
INDEX_PATH = os.path.join(BASE_DIR, "data/vector_db/researchmind.index")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

index = faiss.read_index(INDEX_PATH)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
model.to(device)
model.eval()


def ask_researchmind(question):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, 10)

    context_parts = []

    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        else:
            text = str(chunk)

        if text.strip():
            context_parts.append(text.strip())

    context = "\\n\\n".join(context_parts)

    prompt = f"""
    Answer the question using the research paper context.

    Context:
    {context}

    Question:
    {question}

    Give a simple and direct answer.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if len(answer.split()) < 8:
        return (
            "The research paper context was retrieved, but the model "
            "could not generate a complete answer. Please try another question."
        )

    return answer


def researchmind_interface(question):
    if not question or not question.strip():
        return "Please enter a research question."

    try:
        return ask_researchmind(question)
    except Exception as error:
        return f"Error: {str(error)}"


with gr.Blocks(title="ResearchMind-AI") as demo:

    gr.Markdown("# ResearchMind-AI")
    gr.Markdown("### AI Personal Research Assistant using RAG")
    gr.Markdown(
        "Ask questions about your research papers "
        "and get answers using Retrieval-Augmented Generation."
    )

    question = gr.Textbox(
        label="Ask a Research Question",
        placeholder="Enter your research question..."
    )

    submit_btn = gr.Button("Generate Answer")

    answer = gr.Textbox(
        label="ResearchMind-AI Answer",
        lines=8
    )

    submit_btn.click(
        fn=researchmind_interface,
        inputs=question,
        outputs=answer
    )

if __name__ == "__main__":
    demo.launch()
