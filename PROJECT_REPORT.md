# University Regulation RAG Assistant

## Project Report

### 1. Introduction

University regulations are often distributed across lengthy academic rules, examination guidelines, attendance policies, and student handbooks. Finding one specific rule manually can be slow and confusing. The University Regulation RAG Assistant is a web application that allows students to upload university PDF documents and ask questions about their contents.

The application uses Retrieval-Augmented Generation (RAG). It first searches the uploaded documents for relevant text and then sends only that retrieved text to Google Gemini to generate a grounded answer. This makes the answer more closely connected to the uploaded regulations instead of relying only on the language model's general knowledge.

### 2. Problem Statement

Students need a simple way to find information from multiple university documents. A normal keyword search may miss related wording, while a general chatbot may produce information that is not present in the university documents. This project solves the problem by combining semantic vector search with a grounded language-model response.

### 3. Objectives

- Allow multiple university PDF documents to be uploaded.
- Extract text from every PDF page.
- Preserve the source filename and page number.
- Split long text into smaller searchable chunks.
- Convert document chunks and questions into embeddings.
- Store and search embeddings using FAISS.
- Generate answers using only retrieved document context.
- Display the source PDF and page number for the answer.
- Return a clear message when information is not found.
- Keep indexing separate from question answering so multiple questions can reuse the same index.

### 4. Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Application programming language |
| Streamlit | Web interface |
| PyPDF | PDF text extraction |
| LangChain RecursiveCharacterTextSplitter | Text chunking |
| Sentence Transformers | Document and question embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| Google Gemini API | Grounded answer generation |
| python-dotenv | Loading the API key from `.env` |
| NumPy | Numeric embedding arrays |

### 5. System Architecture

```text
PDF Upload
    |
PDF Page Text Extraction
    |
Text Cleaning
    |
Recursive Character Chunking
    |
Sentence Transformer Embeddings
    |
FAISS Vector Index + Metadata
    |
        User Question
            |
        Question Embedding
            |
        FAISS Similarity Search
            |
        Top K Relevant Chunks
            |
        Gemini with Retrieved Context
            |
        Grounded Answer + Sources
```

### 6. Project Structure

```text
University-RAG-Assistant/
|
|-- app.py
|-- rag.py
|-- requirements.txt
|-- .env
|-- .env.example
|-- .gitignore
|-- README.md
|-- PROJECT_REPORT.md
|
|-- assets/
|-- documents/
|   |-- University_Academic_Regulations.pdf
|   |-- University_Regulations_Sample.pdf
|
|-- vectorstore/
    |-- index.faiss
    |-- metadata.json
```

### 7. Implementation Details

#### 7.1 PDF Loading and Extraction

The `load_pdf()` function reads every page using `PdfReader`. It safely handles invalid files, empty pages, and pages where text extraction fails. Each usable page is stored with metadata:

```python
{
    "text": "Extracted page text",
    "source": "University_Academic_Regulations.pdf",
    "page": 2
}
```

The page number is stored at the beginning of processing so it can later be displayed as a source citation.

#### 7.2 Text Preprocessing

The `clean_text()` function removes empty text and repeated whitespace. It combines unnecessary line breaks and spaces while preserving meaningful words, numbers, and punctuation.

#### 7.3 Text Chunking

Long PDF pages are split using `RecursiveCharacterTextSplitter`. The default values are:

- Chunk size: 500 characters
- Chunk overlap: 50 characters

Chunking is necessary because sending an entire document to a language model is inefficient and may exceed context limits. Overlap helps preserve a sentence or regulation that crosses a chunk boundary.

The chunk metadata is retained:

```python
{
    "text": "A searchable piece of text",
    "source": "University_Academic_Regulations.pdf",
    "page": 2
}
```

#### 7.4 Embeddings

An embedding is a numeric vector that represents the meaning of text. The project uses the `all-MiniLM-L6-v2` Sentence Transformer model.

Both document chunks and user questions are encoded with the same model. This is important because their vectors must exist in the same vector space for meaningful similarity comparison.

#### 7.5 FAISS Similarity Search

FAISS stores document embeddings and quickly searches for vectors that are semantically similar to a question. The project uses a normalized inner-product index, which behaves like cosine similarity search.

The `top_k` setting controls how many relevant chunks are retrieved. The default is 3. A similarity threshold removes very weak matches before Gemini is called.

Metadata is stored separately in the same order as the FAISS vectors. When FAISS returns a vector position, that position is used to retrieve the matching chunk, PDF filename, and page number.

#### 7.6 Gemini Answer Generation

The `generate_answer()` function sends Gemini a prompt containing:

- The user's question
- The retrieved document chunks
- The source and page metadata
- Instructions to use only the supplied context

The prompt tells Gemini not to use outside knowledge, guess, or invent information. The current implementation uses the `google-genai` SDK and reads `GEMINI_API_KEY` from `.env`.

#### 7.7 Grounding and Not-Found Handling

If the FAISS search produces no result above the similarity threshold, the application returns:

> The requested information was not found in the uploaded documents.

Gemini is also instructed to return the same message when the retrieved context does not contain the answer. This reduces hallucination, although no generative AI system can guarantee zero hallucination.

#### 7.8 Source and Page Citations

The `format_sources()` function removes duplicate source/page combinations. The answer displays citations such as:

```text
- University_Academic_Regulations.pdf - Page 2
```

This allows the student to verify the answer in the original document.

### 8. User Interface

The Streamlit interface provides:

- Multiple PDF upload
- Uploaded document list
- Chunk size setting
- Chunk overlap setting
- Top K setting
- Process Documents button
- Question input
- Ask Question button
- Answer section
- Sources and page numbers
- Success, warning, and error messages
- Light and dark mode readable styling
- Campus/library background image with separate theme overlays

The application stores the embedding model, FAISS index, and metadata in Streamlit session state. Therefore, asking another question does not recreate all document embeddings.

### 9. Security

The Gemini API key is never hardcoded in Python. It is loaded from `.env`:

```text
GEMINI_API_KEY=your_real_gemini_api_key
```

The `.env` file is included in `.gitignore` so the secret is not accidentally committed to a public repository. The API key is not displayed in the user interface.

### 10. Error Handling

The application handles the following situations:

- No PDF uploaded
- Invalid PDF
- Empty PDF
- PDF with no extractable text
- No documents processed
- Empty question
- Missing Gemini API key
- Gemini API failure
- FAISS search failure
- Invalid chunk size or overlap

Messages are displayed in simple language through Streamlit alerts.

### 11. Testing and Validation

The project was validated with the following checks:

- Python compilation of `app.py` and `rag.py`
- Editor diagnostics with no reported errors
- PyPDF validation of the sample PDF
- Sample PDF validation result: 3 pages and extractable text
- University regulations PDF validation result: 5 pages and extractable text
- Streamlit page rendering validation
- Light-mode contrast inspection
- Dark-mode CSS treatment added for readable text and controls
- Document processing validation with 24 chunks from two PDFs

### 12. Installation and Execution

Open a terminal in the project folder:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create or update `.env`:

```text
GEMINI_API_KEY=your_real_gemini_api_key
```

Run the application:

```text
streamlit run app.py
```

Open the local URL shown by Streamlit, upload PDFs, click **Process Documents**, and ask questions.

### 13. Sample Questions

1. What is the minimum attendance requirement?
2. What happens if a student has attendance below the required percentage?
3. How are internal assessment marks calculated?
4. What documents are required for medical leave?
5. What are the examination eligibility requirements?
6. What items are prohibited inside the examination hall?
7. What is the process for applying for a re-examination?
8. What is the policy for academic misconduct?
9. What grade is awarded for marks between 75 and 89?
10. How many credits are required for graduation?

### 14. Viva Questions and Answers

**What is RAG?**

RAG means Retrieval-Augmented Generation. It retrieves relevant information from documents and gives that information to a language model before generating an answer.

**Why use RAG?**

RAG helps answer questions using private or specific documents and reduces unsupported answers from general model knowledge.

**What is chunking?**

Chunking divides long document text into smaller pieces that can be searched efficiently and supplied as focused context.

**Why use overlap?**

Overlap preserves context when a sentence or regulation is split at the boundary between two chunks.

**What are embeddings?**

Embeddings are numerical vectors that represent the semantic meaning of text.

**What is FAISS?**

FAISS is a library for efficient similarity search over vector embeddings.

**What is metadata?**

Metadata is additional information attached to text, such as the source PDF filename and page number.

**What is `top_k`?**

`top_k` is the number of most relevant chunks returned from the vector search.

**How are sources identified?**

Each chunk retains its source filename and page number. The FAISS result position maps back to that metadata.

**What happens if the answer is not in the documents?**

The similarity threshold can reject weak search results, and the application displays the information-not-found response instead of sending unsupported context to Gemini.

### 15. Limitations

- Scanned PDFs without a text layer may produce little or no extracted text.
- The quality of the answer depends on PDF extraction and chunking quality.
- A similarity threshold reduces weak retrieval but cannot guarantee perfect retrieval.
- Gemini can still make mistakes, so users should verify important rules in the cited PDF.
- The application uses an in-memory session index and does not provide user accounts or permanent multi-user storage.
- An internet connection and a valid Gemini API key are required for generated answers.

### 16. Future Scope

Possible future improvements include OCR for scanned documents, document deletion controls, persistent per-user indexes, better page previews, authentication, and evaluation using a prepared question-and-answer dataset. These features are outside the scope of this basic RAG project.

### 17. Conclusion

The University Regulation RAG Assistant demonstrates the complete basic RAG workflow: PDF loading, preprocessing, chunking, embedding generation, FAISS vector search, context retrieval, Gemini answer generation, grounding, and source citation. The project is intentionally simple so that each stage can be explained clearly in a college demonstration or viva.
