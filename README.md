# Document Lab: Local RAG Workshop

A classroom-ready application for chatting with PDF and DOCX files. Documents are extracted and embedded on the computer; questions are answered by an installed local Ollama model. No API key or paid service is used.

## RAG in one line

**Document → Extract → Chunk → Embed → Compare → Retrieve → Context → LLM → Answer**

RAG is not the LLM automatically reading a PDF. The application first retrieves relevant text, then passes only that context and the question to the LLM. The embedding model represents text as vectors. Cosine similarity measures the angle between vectors; for normalized vectors the dot product is the cosine score. Similarity finds candidate evidence, while the LLM turns that evidence into a readable answer.

## Requirements and setup

- Python 3.11 recommended.
- Ollama installed and running locally.
- About 2 GB of free disk space for the small embedding model and one quantized chat model; RAM needs vary by model.

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama from [ollama.com/download](https://ollama.com/download), then choose one model. Qwen is preferred; Llama is automatically used when no Qwen model is installed.

```powershell
ollama pull qwen2.5:3b
# Or, if using the fallback family:
ollama pull llama3.2:3b
```

Models are never pulled by the application. Ollama's model tags are inspected at runtime; the first installed Qwen tag wins, otherwise the first Llama tag. If neither exists, the UI shows setup instructions. Smaller models are friendlier to laptops; availability and performance depend on the machine.

Run the application:

```powershell
streamlit run app.py
```

The embedding model is never downloaded by the app. Prepare its local Hugging Face cache explicitly before indexing:

```powershell
hf download sentence-transformers/all-MiniLM-L6-v2
```

After preparation, embedding and inference run locally. For offline classrooms, run this command on each machine before the workshop. If the cache is missing, the UI explains the exact preparation step rather than downloading silently.

## Workshop demonstration

1. Start Ollama and confirm the selected model in the header.
2. Upload a text-based syllabus PDF or DOCX, then choose **Extract, chunk, and embed**.
3. Point out extracted character/page counts, chunk count, and the small configurable overlap.
4. Ask: “What are the eligibility requirements mentioned in this document?”
5. Open **View retrieved context**. Compare chunk IDs, source/page metadata, and cosine scores with the answer.
6. Ask a question whose answer is absent. If every score is below the threshold, no unrelated text is sent to Ollama.
7. Discuss why retrieved document text is reference data and cannot be trusted as instructions.

Other sample questions:

- “When is the application deadline?”
- “Which courses are required in the first semester?”
- “What is the attendance policy?”
- “What is the institution's policy for underwater basket weaving?” (use to demonstrate missing evidence)

## Architecture

```text
app.py (Streamlit UI and session state)
  ├── document_loader.py → text_cleaner.py
  ├── chunker.py → DocumentChunk metadata
  ├── embeddings.py → local Sentence Transformers vectors (CPU)
  ├── retriever.py → explicit NumPy cosine similarity and Top-K
  ├── rag_pipeline.py → threshold, context construction, orchestration
  ├── model_selector.py → Ollama health/model detection, Qwen then Llama
  └── ollama_client.py → local /api/chat with grounded system rules
```

The initial index lives in Streamlit session memory. Clearing the browser session clears that index; source files are not persisted by this application. PDF pages are retained in chunk metadata. DOCX paragraph and table text is retained, but DOCX page numbers are not reliably available and therefore are not invented.

## Configuration and safeguards

- Sidebar controls: selected installed Qwen/Llama model, Top-K, minimum similarity, chunk size, overlap, and generation temperature.
- PDF and DOCX uploads are supported; each file is limited to 25 MB.
- Image-only PDFs need OCR before upload; OCR is deliberately outside this lightweight workshop app.
- Retrieval is visibly separate from generation. If no result passes the threshold, generation is skipped.
- The system instruction tells Ollama to use context only, acknowledge missing evidence, cite sources when possible, and ignore instructions inside document text.
- Uploaded file content is hashed to avoid embedding the same file twice during one session.
- The UI catches extraction, embedding, connection, and generation errors and displays a concise message.

## Tests

```powershell
pytest -q
```

Tests create small PDF/DOCX documents in memory, use a deterministic fake embedder, and test ranking/model selection without a running Ollama server or model downloads.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Ollama not connected | Open the Ollama app/service and confirm `http://localhost:11434` responds. |
| No supported model | Run `ollama pull qwen2.5:3b` or `ollama pull llama3.2:3b`, then refresh. |
| Model is slow or runs out of memory | Choose a smaller quantized model; close memory-heavy applications. |
| No selectable text | The PDF may be scanned/image-only; OCR it before upload. |
| No relevant chunks | Lower the threshold slightly, adjust chunk size, or ask a closer question. |
| Embedding setup fails offline | Cache the Sentence Transformers model before class; model files must be available locally. |
| DOCX pages show unavailable | DOCX pagination depends on word-processor layout; source file and chunk IDs are still shown. |

## Component guide

- `document_loader.py`: reads PDF text page-by-page and DOCX paragraphs/tables.
- `text_cleaner.py`: removes control characters and excess whitespace without discarding punctuation.
- `chunker.py`: creates overlapping bounded text windows with source metadata.
- `embeddings.py`: loads a small Sentence Transformers model on CPU and returns normalized vectors.
- `retriever.py`: demonstrates cosine similarity and ranked, thresholded, de-duplicated Top-K results directly with NumPy.
- `model_selector.py`: probes Ollama and applies the Qwen-first/Llama-fallback policy without pulling models.
- `ollama_client.py`: separates system rules from untrusted context and calls the local chat endpoint.
- `rag_pipeline.py`: stops when retrieval finds no evidence; otherwise assembles source-labeled context and generates the answer.
- `app.py`: classroom UI, index state, progress/error feedback, settings, chat history, and retrieval inspection.