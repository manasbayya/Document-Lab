"""Streamlit interface for the local document RAG workshop."""

from hashlib import sha256
from html import escape

import numpy as np
import streamlit as st

from core.chunker import create_chunks
from core.document_loader import DocumentLoadError, load_document
from core.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingModelUnavailable, EmbeddingService
from core.model_selector import SETUP_MESSAGE, get_model_status
from core.rag_pipeline import answer_question

st.set_page_config(page_title="Document Lab | Local RAG", page_icon="DL", layout="wide")

st.markdown(
    """
    <style>
    :root { --ink:#17221d; --muted:#65746c; --paper:#f4f6f1; --line:#dce3dc; --green:#216e52; }
    html, body, [class*="css"] { font-family:'Segoe UI',sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { padding-top:1.6rem; max-width:1200px; }
    h1,h2,h3 { letter-spacing:0; }
    h1 { font-size:2.15rem; margin-bottom:.2rem; }
    .eyebrow { color:var(--green); font-family:Consolas,monospace; text-transform:uppercase; font-size:.74rem; }
    .subtle { color:var(--muted); }
    .status-strip { border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:.8rem 0; margin:1rem 0 1.4rem; }
    .metric { background:#fff; border:1px solid var(--line); padding:1rem; border-radius:6px; min-height:94px; }
    .metric-label { color:var(--muted); font-size:.76rem; text-transform:uppercase; }
    .metric-value { font-family:Consolas,monospace; font-size:1.3rem; margin-top:.35rem; }
    [data-testid="stSidebar"] { background:#e9eee8; border-right:1px solid var(--line); }
    div[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:6px; }
    .stButton button, .stFormSubmitButton button { border-radius:5px; }
    code, .mono { font-family:Consolas,monospace; }
    @media (max-width:700px) { .block-container { padding:1rem .8rem; } h1 { font-size:1.7rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> EmbeddingService:
    return EmbeddingService(DEFAULT_EMBEDDING_MODEL)


def init_state() -> None:
    defaults = {
        "chunks": [],
        "chunk_vectors": np.empty((0, 0), dtype=np.float32),
        "documents": [],
        "document_hashes": set(),
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


init_state()
status = get_model_status()

with st.sidebar:
    st.markdown("<div class='eyebrow'>Workshop controls</div>", unsafe_allow_html=True)
    st.subheader("Retrieval settings")
    supported_models = [
        model for model in status.models
        if model.split(":", 1)[0].casefold().startswith(("qwen", "llama"))
    ]
    if supported_models:
        preferred_index = supported_models.index(status.selected_model) if status.selected_model in supported_models else 0
        selected_model = st.selectbox("Selected Ollama model", supported_models, index=preferred_index)
    else:
        selected_model = None
        st.caption("No supported model is currently available.")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=8, value=4)
    threshold = st.slider("Minimum cosine similarity", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
    st.divider()
    st.subheader("Chunk settings")
    chunk_size = st.number_input("Chunk size (characters)", min_value=200, max_value=3000, value=900, step=100)
    overlap = st.number_input("Overlap (characters)", min_value=0, max_value=1000, value=120, step=20)
    if overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")
    temperature = st.slider("Model temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    st.caption(f"Embeddings: `{DEFAULT_EMBEDDING_MODEL}` (CPU)")
    st.caption("Uploaded files remain in this browser session's memory.")

left, right = st.columns([1.6, 1], vertical_alignment="center")
with left:
    st.markdown("<div class='eyebrow'>Local retrieval-augmented generation</div>", unsafe_allow_html=True)
    st.title("Document Lab")
    st.markdown("Ask questions about your files. Inspect retrieval before the local model answers.")
with right:
    ollama_label = "Connected" if status.connected else "Not connected"
    model_label = escape(selected_model or "No Qwen/Llama model")
    st.markdown(
        f"<div class='status-strip'><b>Ollama</b> &nbsp; {ollama_label}<br>"
        f"<b>Active model</b> &nbsp; <span class='mono'>{model_label}</span></div>",
        unsafe_allow_html=True,
    )
if not status.connected:
    st.warning("Ollama is not responding. Install it from ollama.com/download if needed, start the Ollama application or service, then refresh this page.")
elif not status.selected_model:
    st.info(SETUP_MESSAGE)

overview = st.columns(4)
with overview[0]:
    metric("Documents", str(len(st.session_state.documents)))
with overview[1]:
    metric("Indexed chunks", str(len(st.session_state.chunks)))
with overview[2]:
    metric("Embeddings", "Ready" if len(st.session_state.chunks) else "Not indexed")
with overview[3]:
    metric("Retrieval", "Ready" if len(st.session_state.chunks) else "Waiting")

st.divider()
upload_col, about_col = st.columns([1.15, 1])
with upload_col:
    st.subheader("1. Add documents")
    uploads = st.file_uploader("Choose PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    if uploads:
        for uploaded in uploads:
            st.caption(f"{uploaded.name} · {uploaded.size / 1024:.1f} KB")
    can_index = bool(uploads) and overlap < chunk_size
    if st.button("Extract, chunk, and embed", type="primary", disabled=not can_index, use_container_width=True):
        try:
            with st.spinner("Extracting text and generating local embeddings..."):
                new_chunks = []
                new_docs = []
                already_indexed = 0
                for uploaded in uploads:
                    if uploaded.size > 25 * 1024 * 1024:
                        st.warning(f"Skipped {uploaded.name}: files are limited to 25 MB for this workshop.")
                        continue
                    content = uploaded.getvalue()
                    document_hash = sha256(content).hexdigest()
                    if document_hash in st.session_state.document_hashes:
                        already_indexed += 1
                        continue
                    document = load_document(uploaded.name, content)
                    chunks = create_chunks(document, int(chunk_size), int(overlap))
                    if not chunks:
                        st.warning(f"No useful text chunks were generated for {uploaded.name}.")
                        continue
                    new_chunks.extend(chunks)
                    new_docs.append(
                        {
                            "name": document.filename,
                            "hash": document_hash,
                            "characters": document.character_count,
                            "pages": sum(page.page_number is not None for page in document.pages),
                            "chunks": len(chunks),
                        }
                    )
                if new_chunks:
                    new_vectors = get_embeddings().encode([chunk.text for chunk in new_chunks])
                    if len(st.session_state.chunks) and st.session_state.chunk_vectors.shape[1] != new_vectors.shape[1]:
                        st.error("Embedding dimensions changed; clear the index and try again.")
                    else:
                        st.session_state.chunks.extend(new_chunks)
                        st.session_state.chunk_vectors = (
                            np.vstack([st.session_state.chunk_vectors, new_vectors])
                            if len(st.session_state.chunks) > len(new_chunks)
                            else new_vectors
                        )
                        st.session_state.documents.extend(new_docs)
                        st.session_state.document_hashes.update(document["hash"] for document in new_docs)
                        st.success(f"Indexed {len(new_docs)} document(s) into {len(new_chunks)} chunks.")
                if already_indexed:
                    st.info(f"Skipped {already_indexed} document(s) already in this session's index.")
        except DocumentLoadError as exc:
            st.error(str(exc))
        except EmbeddingModelUnavailable as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Document processing failed: {exc}")
with about_col:
    st.subheader("The RAG path")
    st.markdown("**Document → Extract → Chunk → Embed → Compare → Retrieve → Context → LLM → Answer**")
    st.caption("Embeddings represent meaning as vectors. Cosine similarity compares vector direction; retrieved text, not the model's prior knowledge, is supplied as answer context.")

if st.session_state.documents:
    with st.expander("Indexed documents"):
        for document in st.session_state.documents:
            page_label = f"{document['pages']} page(s)" if document["pages"] else "DOCX text"
            st.write(f"**{document['name']}** · {document['characters']:,} characters · {page_label} · {document['chunks']} chunks")
    if st.button("Clear index and conversation"):
        st.session_state.chunks = []
        st.session_state.chunk_vectors = np.empty((0, 0), dtype=np.float32)
        st.session_state.documents = []
        st.session_state.document_hashes = set()
        st.session_state.messages = []
        st.rerun()

st.divider()
st.subheader("2. Chat with your documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View retrieved context"):
                for source in message["sources"]:
                    chunk = source.chunk
                    page = f" · page {chunk.page_number}" if chunk.page_number else ""
                    st.markdown(f"**{chunk.chunk_id}** · {source.score:.3f} · {chunk.source}{page}")
                    st.write(chunk.text)

question = st.chat_input("Ask a question about the indexed documents", disabled=not st.session_state.chunks)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    if not status.connected:
        response = "Ollama is not connected. Start the local Ollama service, then retry your question."
        sources = []
    elif not selected_model:
        response = SETUP_MESSAGE
        sources = []
    else:
        try:
            result = answer_question(
                question,
                st.session_state.chunks,
                st.session_state.chunk_vectors,
                get_embeddings(),
                selected_model,
                top_k=int(top_k),
                threshold=float(threshold),
                temperature=float(temperature),
            )
            response, sources = result.answer, result.sources
        except Exception as exc:
            response, sources = f"I couldn't complete the local answer: {exc}", []
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
    st.rerun()
elif not st.session_state.chunks:
    st.info("Index at least one PDF or DOCX to start asking questions.")

st.divider()
st.caption("Uploaded documents may contain misleading or malicious instructions. Retrieved text is treated as reference material, not as commands. No paid APIs or API keys are used.")