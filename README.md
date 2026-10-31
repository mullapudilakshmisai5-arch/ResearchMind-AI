# ResearchMind-AI
## AI Personal Research Assistant using Retrieval-Augmented Generation

ResearchMind-AI is an AI-powered research assistant designed to
retrieve information from research papers and generate answers
using Retrieval-Augmented Generation (RAG).

## Features

- Research paper PDF text extraction
- Text cleaning and document chunking
- Semantic embeddings generation
- FAISS vector database
- Relevant research information retrieval
- Context-based answer generation
- Source-level citations support
- Research evaluation
- Gradio web interface

## Technologies Used

- Python
- Google Colab
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- FLAN-T5
- Gradio
- NumPy

## Project Architecture

PDF Research Papers
        |
        v
Text Extraction
        |
        v
Text Cleaning and Chunking
        |
        v
Embedding Generation
        |
        v
FAISS Vector Database
        |
        v
Semantic Retrieval
        |
        v
Context-based Answer Generation
        |
        v
Gradio Interface

## Project Structure

ResearchMind-AI/
|
|-- data/
|   |-- papers/
|   |-- processed/
|   |-- vector_db/
|
|-- src/
|   |-- config.py
|   |-- retrieval.py
|   |-- rag_pipeline.py
|   |-- citations.py
|   |-- research_evaluation.py
|   |-- research_intelligence.py
|   |-- phase5_retrieval.py
|
|-- app/
|   |-- app.py
|
|-- notebooks/
|
|-- requirements.txt
|-- README.md
|-- .gitignore

## Installation

Clone the repository:

    git clone https://github.com/mullapudilakshmisai5-arch/ResearchMind-AI.git

Navigate to the project directory:

    cd ResearchMind-AI

Install dependencies:

    pip install -r requirements.txt

## Running the Application

Run the Gradio application from the project root:

    python app/app.py

The application requires the processed research chunks and
FAISS index to be available in the expected data directories.

## Project Goals

1. Search information across research documents.
2. Answer research-related questions.
3. Provide source-level references.
4. Summarize research papers.
5. Compare multiple research papers.
6. Identify potential research gaps.

## Current Limitations

- Answer quality depends on the language model.
- Generated answers require evaluation for accuracy.
- Research documents and vector indexes must be available.
- Some answer-generation tests use a rule-based fallback
  when the language model output is insufficient.

## Author

Lakshmi Sai

AI & Machine Learning Student

## Project Status

ResearchMind-AI is developed as an AI/ML portfolio project.
The implementation includes document processing, semantic
retrieval, answer generation, evaluation, and a Gradio interface.
