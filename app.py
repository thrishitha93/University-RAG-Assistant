import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    NOT_FOUND_MESSAGE,
    create_faiss_index,
    create_chunks,
    format_sources,
    generate_answer,
    load_pdf,
    save_vectorstore,
    search_documents,
)

load_dotenv()

st.set_page_config(page_title="University Regulation RAG Assistant", page_icon="book", layout="wide")

background_image = Path(__file__).parent / "assets" / "campus.jpg"
background_source = (
    f"url('{background_image.as_posix()}')"
    if background_image.exists()
    else "url('https://images.unsplash.com/photo-1564981797816-1043664bf78d?auto=format&fit=crop&w=2200&q=85')"
)
st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: light;
        --page-text: #17324d;
        --page-muted: #38566f;
        --panel: rgba(255, 255, 255, .94);
        --panel-border: rgba(255, 255, 255, .78);
        --field: #ffffff;
        --field-border: #829ab1;
        --overlay: linear-gradient(90deg, rgba(255, 255, 255, .93) 0%, rgba(244, 249, 251, .78) 48%, rgba(224, 236, 241, .58) 100%);
    }}
    [data-testid="stAppViewContainer"] {{ background-image: var(--overlay), {background_source}; background-position: center; background-size: cover; background-attachment: fixed; color: var(--page-text); }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background: rgba(247, 250, 252, .98); color: var(--page-text); border-right: 1px solid #d9e2ec; }}
    [data-testid="stSidebar"] * {{ color: var(--page-text); }}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {{ color: var(--page-text) !important; font-weight: 600; }}
    [data-testid="stSidebar"] input {{ background: var(--field); color: var(--page-text); border: 1px solid var(--field-border); }}
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"], [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * {{ color: #ffffff !important; }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background: var(--field); border: 1px dashed var(--field-border); }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{ color: var(--page-text) !important; }}
    .main .block-container {{ max-width: 1100px; padding: 3rem 2rem 5rem; }}
    .hero {{ max-width: 760px; padding: 2rem 0 1.4rem; color: var(--page-text); }}
    .hero h1 {{ font-family: Georgia, serif; font-size: clamp(2.4rem, 6vw, 4.8rem); line-height: 1.05; margin: 0; letter-spacing: 0; overflow-wrap: anywhere; }}
    .hero p {{ font-size: clamp(1rem, 2vw, 1.12rem); line-height: 1.6; max-width: 650px; color: var(--page-muted); margin-top: 1.2rem; }}
    .main [data-testid="stWidgetLabel"] p, .main .stMarkdown p, .main label, .main h2, .main h3 {{ color: var(--page-text) !important; }}
    .main input {{ background: var(--field); color: var(--page-text); border: 1px solid var(--field-border); }}
    .main input::placeholder {{ color: #526d82; opacity: 1; }}
    .main [data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--panel); border: 1px solid var(--panel-border); border-radius: 14px; padding: .8rem 1rem; box-shadow: 0 10px 30px rgba(16, 42, 67, .12); }}
    .main [data-testid="stAlert"] {{ background: var(--panel); color: var(--page-text); border: 1px solid var(--panel-border); }}
    .main [data-testid="stAlert"] * {{ color: var(--page-text) !important; }}
    .main [data-testid="stBaseButton-primary"], .main [data-testid="stBaseButton-primary"] * {{ color: #ffffff !important; }}
    html[data-theme="dark"], [data-theme="dark"] {{ color-scheme: dark; --page-text: #eef6fb; --page-muted: #d5e4ec; --panel: rgba(14, 29, 43, .94); --panel-border: rgba(173, 204, 219, .32); --field: #f8fbfd; --field-border: #b6ccd8; --overlay: linear-gradient(90deg, rgba(4, 16, 27, .91) 0%, rgba(4, 16, 27, .78) 48%, rgba(4, 16, 27, .62) 100%); }}
    html[data-theme="dark"] [data-testid="stSidebar"], [data-theme="dark"] [data-testid="stSidebar"] {{ background: rgba(10, 25, 38, .98); color: var(--page-text); border-right-color: #29465a; }}
    html[data-theme="dark"] [data-testid="stSidebar"] *, [data-theme="dark"] [data-testid="stSidebar"] * {{ color: var(--page-text); }}
    html[data-theme="dark"] [data-testid="stSidebar"] input, [data-theme="dark"] [data-testid="stSidebar"] input {{ background: #172f42; color: var(--page-text); border-color: #628397; }}
    html[data-theme="dark"] [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"], [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background: #172f42; border-color: #81a6b9; }}
    html[data-theme="dark"] [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *, [data-theme="dark"] [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{ color: var(--page-text) !important; }}
    html[data-theme="dark"] .main input, [data-theme="dark"] .main input {{ color: #102a43; }}
    html[data-theme="dark"] .main input::placeholder, [data-theme="dark"] .main input::placeholder {{ color: #526d82; }}
    @media (max-width: 640px) {{
        .main .block-container {{ padding: 2rem 1rem 4rem; }}
        .hero {{ padding-top: 1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>University Regulation<br>RAG Assistant</h1><p>Find clear, source-backed answers across your academic regulations, examination guidelines, attendance policies, and student handbooks.</p></div>',
    unsafe_allow_html=True,
)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.embedding_model = None
    st.session_state.document_names = []

with st.sidebar:
    st.header("Document setup")
    uploaded_files = st.file_uploader("Upload university PDFs", type=["pdf"], accept_multiple_files=True)
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, step=10)
    top_k = st.number_input("Top K", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)
    process_documents = st.button("Process Documents", type="primary", use_container_width=True)

if uploaded_files:
    st.subheader("Uploaded documents")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
else:
    st.info("Upload one or more PDF documents in the sidebar to begin.")

if process_documents:
    if not uploaded_files:
        st.error("Please upload at least one PDF before processing.")
    else:
        try:
            pages = []
            unusable_files = []
            for uploaded_file in uploaded_files:
                file_pages = load_pdf(uploaded_file)
                if file_pages:
                    pages.extend(file_pages)
                else:
                    unusable_files.append(uploaded_file.name)
            chunks = create_chunks(pages, int(chunk_size), int(chunk_overlap))
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            index, metadata = create_faiss_index(chunks, embedding_model)
            st.session_state.index = index
            st.session_state.chunks = metadata
            st.session_state.embedding_model = embedding_model
            st.session_state.document_names = [file.name for file in uploaded_files]
            save_vectorstore(index, metadata)
            st.success(f"Processed {len(metadata)} chunks from {len(uploaded_files)} PDF(s). You can now ask multiple questions without rebuilding embeddings.")
            if unusable_files:
                st.warning("No extractable text was found in: " + ", ".join(unusable_files))
        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Document processing failed: {error}")

with st.container(border=True):
    st.subheader("Ask a question")
    question = st.text_input("Question", placeholder="For example: What is the minimum attendance requirement?")
    ask_question = st.button("Ask Question", type="primary")

if ask_question:
    if not question.strip():
        st.error("Please enter a question.")
    elif st.session_state.index is None:
        st.error("Please process your PDF documents before asking a question.")
    else:
        try:
            with st.spinner("Searching the uploaded documents..."):
                retrieved_chunks = search_documents(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.embedding_model,
                    int(top_k),
                    DEFAULT_SIMILARITY_THRESHOLD,
                )
                answer = generate_answer(question, retrieved_chunks, os.getenv("GEMINI_API_KEY", ""))
            with st.container(border=True):
                st.subheader("Answer")
                st.write(answer)
                if answer != NOT_FOUND_MESSAGE and retrieved_chunks:
                    st.subheader("Sources")
                    for source in format_sources(retrieved_chunks):
                        st.write(f"- {source}")
        except ValueError as error:
            st.error(str(error))
        except RuntimeError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Question answering failed: {error}")
