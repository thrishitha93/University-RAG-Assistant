# University Regulation RAG Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) app for asking questions about uploaded university PDF documents.

## Files

- `app.py`: Streamlit interface, document-processing button, question form, and session state.
- `rag.py`: PDF extraction, preprocessing, chunking, embeddings, FAISS search, Gemini generation, and source formatting.
- `requirements.txt`: Python packages used by the app.
- `.env.example`: Template for the Gemini key.
- `.env`: Local key file. It is ignored by Git and must never contain a committed real key.
- `documents/`: Optional place to keep local PDFs; uploads are selected in the app.
- `vectorstore/`: Stores the generated FAISS index and JSON metadata after processing.

## Install and run

Open a terminal in this folder:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if needed, then replace the placeholder:

```text
GEMINI_API_KEY=your_real_gemini_api_key
```

Start the app:

```text
streamlit run app.py
```

Upload one or more PDFs, choose chunk settings, click **Process Documents**, and then ask multiple questions. The embedding model and FAISS index are kept in Streamlit session state, so they are not rebuilt for every question.

## How the basic RAG pipeline works

1. `load_pdf()` reads every PDF page with PyPDF and records `source` and `page` metadata.
2. `clean_text()` removes repeated whitespace without removing meaningful words.
3. `create_chunks()` uses `RecursiveCharacterTextSplitter` with configurable size and overlap. Chunks keep page metadata. Overlap helps preserve rules that cross a chunk boundary.
4. `create_embeddings()` converts chunks into vectors with `all-MiniLM-L6-v2`. The same model encodes questions so both documents and queries share one vector space.
5. `create_faiss_index()` stores normalized chunk vectors in a FAISS inner-product index, which acts as cosine similarity search.
6. `search_documents()` embeds the question and retrieves the configurable top K results. Weak matches below the basic similarity threshold are discarded.
7. `generate_answer()` sends only retrieved chunks and the question to Gemini, with instructions not to use outside knowledge.
8. `format_sources()` removes duplicate source/page pairs and displays the page numbers stored during extraction.

If no chunks pass the similarity threshold, the app returns: `The requested information was not found in the uploaded documents.` This reduces hallucination, but no generative system can guarantee zero hallucination.

## Viva concepts

RAG combines retrieval with generation: relevant document text is retrieved first, then a language model produces an answer grounded in that text. Document loading reads source files; preprocessing cleans extraction noise; chunking makes large documents searchable; embeddings represent meaning as numbers; semantic search finds related meaning rather than only exact words; FAISS is the vector index; metadata identifies the source and page; `top_k` controls how many chunks are sent to Gemini; grounding means restricting the answer to retrieved context; hallucination means unsupported generated content.

## Sample questions

1. What is the minimum attendance requirement?
2. What happens if a student falls below the attendance requirement?
3. How are examination marks calculated?
4. What is the procedure for applying for a re-examination?
5. When can a student be debarred from an examination?
6. What are the rules for internal assessment?
7. How many credits are required for graduation?
8. What is the policy for academic misconduct?
9. How can a student apply for leave?
10. What is the process for resolving a mark discrepancy?
