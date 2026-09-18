"""Basic RAG pipeline for university PDF documents.

The module deliberately keeps indexing separate from question answering:
1. Indexing: PDFs -> cleaned page text -> chunks -> embeddings -> FAISS.
2. Question answering: question -> embedding -> FAISS search -> Gemini.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.35
NOT_FOUND_MESSAGE = "The requested information was not found in the uploaded documents."


# A page-level record keeps the source and page attached to its text.
def load_pdf(pdf_file: Any) -> list[dict[str, Any]]:
    """Extract text from every page of an uploaded PDF."""
    try:
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)
    except Exception as error:
        raise ValueError(f"Could not read '{getattr(pdf_file, 'name', 'PDF')}' as a PDF.") from error

    pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        cleaned = clean_text(text)
        if cleaned:
            pages.append({"text": cleaned, "source": pdf_file.name, "page": page_number})
    return pages


def clean_text(text: str) -> str:
    """Remove extraction noise while preserving the words and punctuation."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def create_chunks(
    pages: list[dict[str, Any]], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[dict[str, Any]]:
    """Split pages into searchable pieces while copying page metadata.

    Chunking gives the model focused context instead of an entire book. Overlap
    keeps a sentence or rule from being lost when it crosses a chunk boundary.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be at least zero and smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[dict[str, Any]] = []
    for page in pages:
        for text in splitter.split_text(page["text"]):
            chunks.append({"text": text, "source": page["source"], "page": page["page"]})
    return chunks


def create_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Turn text into vectors; the same model must encode documents and queries."""
    if not texts:
        return np.empty((0, 384), dtype="float32")
    # An embedding is a numeric representation of meaning. Using one model for
    # both sides puts document chunks and questions in the same vector space.
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")


def create_faiss_index(chunks: list[dict[str, Any]], model: SentenceTransformer) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Create a cosine-similarity FAISS index and its parallel metadata list."""
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded PDFs.")
    embeddings = create_embeddings([chunk["text"] for chunk in chunks], model)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def search_documents(
    question: str,
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return relevant chunks, filtering weak cosine-similarity matches."""
    if not question.strip():
        return []
    if top_k <= 0:
        raise ValueError("Top K must be greater than zero.")
    question_embedding = create_embeddings([question], model)
    scores, positions = index.search(question_embedding, min(top_k, len(chunks)))
    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position >= 0 and float(score) >= similarity_threshold:
            result = dict(chunks[position])
            result["score"] = float(score)
            results.append(result)
    return results


def generate_answer(question: str, retrieved_chunks: list[dict[str, Any]], api_key: str, model_name: str = "gemini-3.6-flash") -> str:
    """Ask Gemini for a grounded answer using only retrieved context."""
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Gemini API key is missing. Add GEMINI_API_KEY to your .env file.")
    if not retrieved_chunks:
        return NOT_FOUND_MESSAGE

    context = "\n\n".join(
        f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )
    prompt = f"""You are a University Regulation Assistant.
Answer the user's question ONLY using the provided context from uploaded university documents.
Do not use outside knowledge. Do not guess or invent information.
If the answer cannot be found in the context, reply exactly:
{NOT_FOUND_MESSAGE}

Context:
{context}

User question: {question}
"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        answer = (response.text or "").strip()
        return answer or NOT_FOUND_MESSAGE
    except Exception as error:
        raise RuntimeError(f"Gemini could not generate an answer: {error}") from error


def format_sources(retrieved_chunks: list[dict[str, Any]]) -> list[str]:
    """Return unique source/page citations in retrieval order."""
    unique_sources: list[str] = []
    seen: set[tuple[str, int]] = set()
    for chunk in retrieved_chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(f"{chunk['source']} - Page {chunk['page']}")
    return unique_sources


def save_vectorstore(index: faiss.Index, chunks: list[dict[str, Any]], folder: str = "vectorstore") -> None:
    """Save the index and metadata so the result is inspectable and reusable."""
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(destination / "index.faiss"))
    (destination / "metadata.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
