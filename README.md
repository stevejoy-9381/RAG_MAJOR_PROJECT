# 🧠 DocMind — Enterprise-Grade Multi-User RAG Document Intelligence System

A state-of-the-art, multi-tenant Retrieval-Augmented Generation (RAG) system with JWT authentication, per-user document isolation, hybrid dense/sparse retrieval (FAISS + BM25), CrossEncoder neural re-ranking, query rewriting/expansion, dual LLM provider routing (Local Ollama & Cloud Groq), voice input/output, local model benchmarking, and full observability analytics.

---

## 📋 Table of Contents

1. [System Architecture & Overview](#-system-architecture--overview)
2. [Key Features](#-key-features)
3. [End-to-End Pipeline & Data Flow](#-end-to-end-pipeline--data-flow)
   - [Document Ingestion Pipeline](#1-document-ingestion-pipeline)
   - [Retrieval & Neural Re-ranking](#2-retrieval--neural-re-ranking)
   - [Query Rewriting & Expansion](#3-query-rewriting--expansion)
   - [Dual LLM Routing & Streaming](#4-dual-llm-routing--streaming)
   - [State Management & Memory](#5-state-management--memory)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Supported Document Formats & OCR](#-supported-document-formats--ocr)
7. [Multilingual Retrieval & Embedding Models](#-multilingual-retrieval--embedding-models)
8. [Configuration & Environment Variables](#-configuration--environment-variables)
9. [Installation & Setup](#-installation--setup)
   - [Prerequisites](#prerequisites)
   - [Backend Setup](#backend-setup)
   - [Frontend Setup](#frontend-setup)
   - [Docker & Docker Compose](#docker--docker-compose)
10. [API Reference & Endpoints](#-api-reference--endpoints)
    - [Authentication](#authentication-endpoints)
    - [Documents & Upload](#documents--upload-endpoints)
    - [Chat & Streaming](#chat--streaming-endpoints)
    - [Conversations](#conversation-management-endpoints)
    - [Benchmarking](#benchmarking-endpoints)
    - [Analytics & System Health](#analytics--system-health-endpoints)
11. [Frontend Application Features](#-frontend-application-features)
    - [Authentication Flow](#authentication--session-guard)
    - [Document Management & Library](#document-management--library)
    - [Chat Interface & Voice Interaction](#chat-interface--voice-interaction)
    - [Offline / Online Provider Switch](#offline--online-provider-switch)
    - [Model Benchmarking UI](#model-benchmarking-ui)
    - [Analytics Dashboard](#analytics-dashboard)
12. [Local Model Benchmarking Suite](#-local-model-benchmarking-suite)
13. [Troubleshooting & FAQ](#-troubleshooting--faq)
14. [License](#-license)

---

## 🏗️ System Architecture & Overview

DocMind is architected as a decoupled, microservices-ready full-stack application:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          NEXT.JS 14 FRONTEND                           │
│  (Tailwind CSS, Lucide Icons, Recharts, Web Speech API, React Markdown) │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP / Server-Sent Events (SSE)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND (Python)                        │
│ ┌──────────────────────┐ ┌───────────────────┐ ┌─────────────────────┐ │
│ │   Auth (JWT+Bcrypt)  │ │ Document Store    │ │ Chat Memory & SQLite│ │
│ └──────────────────────┘ └───────────────────┘ └─────────────────────┘ │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │                    Retrieval & Ingestion Engine                    │ │
│ │  - PyMuPDF / python-docx / python-pptx / pandas / Tesseract OCR    │ │
│ │  - SentenceTransformers Embedding Singleton (CPU Thread-Optimized) │ │
│ │  - SQLite Embedding Cache (WAL mode, SHA-256 content hashes)       │ │
│ │  - Per-User Isolated FAISS Vectorstore + BM25 Sparse Index         │ │
│ │  - CrossEncoder Neural Re-ranker (ms-marco-MiniLM-L-6-v2)          │ │
│ │  - Multi-Query Rewriter & Fan-out Deduplicator                     │ │
│ └────────────────────────────────────────────────────────────────────┘ │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │                    LLM Provider Routing Layer                      │ │
│ │  - Auto / Online (Groq: Qwen 2.5 32B / LLaMA 3.3 70B / Mixtral)    │ │
│ │  - Offline (Ollama: LLaMA 3.1 8B / Mistral / DeepSeek / Gemma)     │ │
│ │  - Exponential Backoff Retry & Failover Support                    │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Multi-Tenant Document Isolation**: Every user has an isolated vector store (`vectorstore/{user_id}/faiss_index`) and private document metadata.
- **JWT Authentication & Security**: Complete user registration, bcrypt password hashing, OAuth2 Bearer token verification, and path traversal protection.
- **Hybrid Retrieval Architecture**: Combines semantic dense vector search (FAISS) with keyword-based sparse search (BM25) via weighted reciprocal rank/ensemble scoring (`BM25_WEIGHT=0.4`, `FAISS_WEIGHT=0.6`).
- **Neural Re-ranking**: CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores candidate documents by joint query-document relevance.
- **Query Rewriting & Expansion**: Generates sub-queries, alternative formulations, and translated queries to improve retrieval recall.
- **Dual LLM Providers with Auto-Failover**:
  - **Online**: Groq Cloud API for ultra-low latency response generation.
  - **Offline / Local**: Ollama integration for air-gapped or private inference.
  - **Auto**: Automatically routes to the best available provider with graceful fallback.
- **Real-Time Streaming**: Token-by-token Server-Sent Events (SSE) streaming with optional reasoning/thinking token capture (`<think>` blocks).
- **Persistent SQLite Chat History**: Full multi-session conversation management with automatic title generation, rename, delete, and sliding window memory.
- **Voice Input & Text-to-Speech Output**: Built-in speech recognition for voice prompts and neural speech synthesis for audio playback.
- **Comprehensive Document Support**: Ingests `.pdf` (with OCR fallback for scanned pages), `.docx`, `.pptx`, `.txt`, `.csv`, `.xlsx`, and `.xls`.
- **High-Performance Ingestion**: Tabular data chunking, PyTorch thread tuning, SHA-256 embedding deduplication, and SQLite WAL embedding caching.
- **Local Model Benchmarking Suite**: Side-by-side comparison of local Ollama models across latency, throughput (tokens/sec), answer quality, and retrieval grounding.
- **Observability & Analytics**: Query latency breakdown, token consumption tracking, cache hit rates, document library statistics, and error diagnostics.

---

## 🔄 End-to-End Pipeline & Data Flow

### 1. Document Ingestion Pipeline
1. **File Upload & Validation**: Validates file extension, sanitizes filename (path traversal defense), and enforces upload size limits (`MAX_UPLOAD_SIZE_BYTES=100MB`).
2. **Text Extraction**:
   - **PDF**: PyMuPDF (`fitz`) parses text layers; if pages are image-only scans, Tesseract OCR extracts page text.
   - **Office (DOCX / PPTX)**: Extracted via `python-docx` and `python-pptx`.
   - **Tabular (CSV / Excel)**: Parsed with `pandas` into row-aligned tabular markdown blocks (`TABULAR_ROWS_PER_CHUNK=200`).
   - **Plain Text**: Direct UTF-8 decoding with fallback encoding recovery.
3. **Chunking**: `RecursiveCharacterTextSplitter` chunks text using `CHUNK_SIZE=800` and `CHUNK_OVERLAP=100` with semantic boundary separators (`\n\n`, `\n`, ` `, `""`).
4. **Deduplication & Embedding Cache**: Each chunk is hashed with SHA-256. If the hash exists in `vectorstore/embeddings_cache.db`, the cached vector is reused; otherwise, the singleton `HuggingFaceEmbeddings` model generates vector embeddings in batches (`EMBEDDING_BATCH_SIZE=256`).
5. **Vector Indexing**: Chunks are added incrementally to the user's FAISS index and written to disk (`vectorstore/{user_id}/faiss_index/index.faiss`).

### 2. Retrieval & Neural Re-ranking
1. **Hybrid Retrieval**:
   - **FAISS**: Retrieves top-$K$ candidate chunks using cosine / L2 vector similarity.
   - **BM25**: Retrieves top-$K$ chunks based on term frequencies and inverse document frequencies.
   - **Ensemble**: Combines scores using weighted convex blending (`BM25_WEIGHT=0.4`, `FAISS_WEIGHT=0.6`).
2. **CrossEncoder Re-ranking**: The top candidate chunks (e.g., 10 candidates) are paired with the query `(query, passage)` and scored through `cross-encoder/ms-marco-MiniLM-L-6-v2`. The highest scoring chunks (`FINAL_CONTEXT_K=4`) are selected for prompt augmentation.

### 3. Query Rewriting & Expansion
When enabled (`QUERY_REWRITING_ENABLED=true` or requested via API parameter):
- The LLM receives the user prompt and outputs 3 search-optimized variations, synonyms, and sub-queries.
- Multi-query fan-out retrieves candidate documents for all query variants.
- SHA-256 content deduplication prevents redundant context chunks before re-ranking.

### 4. Dual LLM Routing & Streaming
- When the user sends a message, `get_provider(llm_mode)` selects the appropriate backend (`GroqProvider` or `OllamaProvider`).
- Messages are augmented with the strict system prompt, context chunks with source metadata, and conversation history.
- The response is streamed via SSE (`data: {"type": "token", "content": "..."}`) or returned synchronously.

### 5. State Management & Memory
- **In-Memory Buffer**: Sliding window history for low-latency prompt construction.
- **Persistent SQLite Store**: `vectorstore/chat_history.db` stores conversation metadata, individual user/assistant messages, cited sources, provider tags, and generation latencies.

---

## 💻 Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.10+ / 3.11+)
- **ASGI Server**: Uvicorn
- **Vector Indexing**: FAISS (`faiss-cpu`)
- **Sparse Search**: LangChain BM25 Retriever (`rank_bm25`)
- **Embeddings & Neural Models**: Sentence-Transformers, HuggingFace Embeddings, CrossEncoder
- **LLM APIs**: Groq SDK (`groq`), Ollama HTTP API (`httpx`)
- **Document Parsers**: PyMuPDF (`fitz`), `python-docx`, `python-pptx`, `pandas`, `openpyxl`, `pytesseract`
- **Security**: PyJWT, passlib with bcrypt
- **Storage**: SQLite3 with Write-Ahead Logging (WAL)

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript (v5)
- **Styling**: Tailwind CSS, PostCSS, Lucide React Icons
- **Data Visualization**: Recharts
- **Markdown & Code Highlighting**: `react-markdown`, `remark-gfm`
- **Audio & Speech**: Web Speech API (`SpeechRecognition`, `SpeechSynthesis`)
- **Networking**: Axios, Fetch SSE streaming

---

## 📁 Project Structure

```
RAG-PROJECT-combined/
├── README.md                      # Complete Project Documentation (this file)
└── merged/
    ├── backend/
    │   ├── api.py                 # FastAPI Application & REST/SSE Endpoints
    │   ├── rebuild_index.py       # Vector Index Maintenance & Model Migration Tool
    │   ├── requirements.txt       # Full Backend Dependencies
    │   ├── requirements.api.txt   # Minimal Production Dependencies
    │   ├── Dockerfile             # Multi-stage Backend Container Image
    │   ├── Dockerfile.api         # Lightweight API Container Image
    │   ├── docker-compose.yml     # Local Full-Stack Docker Compose Setup
    │   ├── docker-compose.prod.yml# Production Container Setup
    │   ├── render.yaml            # Cloud Deployment Blueprint
    │   ├── .env.example           # Backend Environment Configuration Template
    │   ├── .env                   # Local Backend Configuration (Secret)
    │   ├── src/
    │   │   ├── __init__.py
    │   │   ├── config.py          # Central Configuration & Environment Parameters
    │   │   ├── auth.py            # JWT Token Creation, Decoding & Bcrypt Hashing
    │   │   ├── user_store.py      # User Registry & Account Persistence (SQLite/JSON)
    │   │   ├── ingest.py          # Document Ingestion, OCR & Batched Embedding Engine
    │   │   ├── retriever.py       # FAISS + BM25 Hybrid Retriever & CrossEncoder
    │   │   ├── query_rewriter.py  # LLM Query Expansion & Rewriting
    │   │   ├── llm_provider.py    # Dual Provider Routing (Ollama & Groq)
    │   │   ├── llm.py             # Prompt Templates & LangChain LLM Helpers
    │   │   ├── chat_memory.py     # In-Memory Buffer & Sliding Window Conversation
    │   │   ├── conversation_store.py # Persistent SQLite Conversation & Message Store
    │   │   ├── document_store.py  # Document Library Metadata Registry
    │   │   ├── benchmark.py       # Automated Ollama Multi-Model Benchmarking Engine
    │   │   ├── analytics.py       # Metrics Aggregator (Latency, Tokens, Cache)
    │   │   ├── logger.py          # Structured Logger & Execution Timers
    │   │   └── utils/
    │   │       └── retry.py       # Exponential Backoff Retry Decorators
    │   └── tests/                 # Unit & Integration Test Suites
    │       ├── test_analytics.py
    │       ├── test_benchmark.py
    │       ├── test_full_system.py
    │       ├── test_multilingual.py
    │       ├── test_query_rewriting.py
    │       └── test_reranking.py
    │
    └── frontend/
        ├── package.json           # Node Dependencies & Build Scripts
        ├── tsconfig.json          # TypeScript Configuration
        ├── tailwind.config.ts     # Tailwind CSS Theme & Styling Config
        ├── next.config.mjs        # Next.js Application Settings
        ├── .env.local.example     # Frontend Environment Template
        ├── .env.local             # Frontend Local Environment Config
        ├── app/
        │   ├── layout.tsx         # Root HTML Layout & Global Font Provider
        │   ├── page.tsx           # Home Redirection / Landing Page
        │   ├── globals.css        # Global CSS & Glassmorphism Utilities
        │   ├── auth/
        │   │   └── page.tsx       # Login & Registration Authentication Page
        │   ├── chat/
        │   │   └── page.tsx       # Main Interactive Chat & Document Intelligence UI
        │   ├── benchmark/
        │   │   └── page.tsx       # LLM Model Benchmarking Dashboard
        │   └── analytics/
        │       └── page.tsx       # Performance & Usage Telemetry Dashboard
        ├── components/
        │   ├── Sidebar.tsx        # Navigation, User Profile, Document Upload & List
        │   ├── ChatMessage.tsx    # Message Rendering, Markdown, Citations & TTS
        │   ├── SourcePanel.tsx    # Cited Document Drawer & Chunk Viewer
        │   ├── LLMModeToggle.tsx  # Mode Switcher (Auto / Online Groq / Offline Ollama)
        │   └── RecentsPanel.tsx   # Chat Session History & Conversation Search
        ├── lib/
        │   ├── api.ts             # Axios API Client & SSE Streaming Handler
        │   ├── auth.ts            # LocalStorage Token Management & Auth Hooks
        │   └── speech.ts          # Web Speech Synthesis & Recognition Wrapper
        └── types/                 # TypeScript Shared Interfaces & Data Types
```

---

## 📄 Supported Document Formats & OCR

| File Format | Extension | Parser Engine | Extraction Details |
| :--- | :--- | :--- | :--- |
| **Portable Document Format** | `.pdf` | PyMuPDF (`fitz`) | High-speed text and layout extraction. |
| **Scanned PDF (Image-only)** | `.pdf` | Tesseract OCR | Automatic fallback OCR for non-text scanned pages. |
| **Microsoft Word** | `.docx` | `python-docx` | Extracts paragraphs, headers, and bulleted lists. |
| **Microsoft PowerPoint** | `.pptx` | `python-pptx` | Extracts slide titles, text frames, and notes. |
| **Plain Text / Code** | `.txt` | Native Python IO | Multi-encoding UTF-8 / Latin-1 text ingestion. |
| **Comma-Separated Values** | `.csv` | `pandas` | Formatted into structured tabular text chunks. |
| **Microsoft Excel** | `.xlsx`, `.xls` | `pandas` + `openpyxl` | Ingests multi-sheet tables with column headers. |

### OCR Setup (Optional for scanned image-only PDFs)
If you are processing scanned PDFs that lack an embedded text layer:
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to system `PATH`.
- **Docker**: Included automatically in both Dockerfiles.

---

## 🌐 Multilingual Retrieval & Embedding Models

DocMind supports cross-lingual Q&A (e.g., asking questions in Hindi, Telugu, Spanish, German, or Japanese against English documents).

To switch to multilingual vector embeddings:
1. Update `EMBEDDING_MODEL` in `backend/.env`:
   ```env
   EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```
2. Reset and rebuild existing vector indices using the migration utility:
   ```bash
   python rebuild_index.py --all
   ```
3. Re-upload your documents. The system will retrieve passages across language boundaries and respond in the language of the query.

---

## ⚙️ Configuration & Environment Variables

### Backend (`merged/backend/.env`)

```env
# ── Authentication ──
JWT_SECRET_KEY=your-secure-random-secret-key-here-minimum-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080  # 7 Days

# ── Embedding Model ──
# English optimized (Default): sentence-transformers/all-MiniLM-L6-v2
# Multilingual (50+ languages): sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE=256

# ── Neural Re-ranker ──
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKING_ENABLED=true

# ── Chunking & Retrieval Parameters ──
CHUNK_SIZE=800
CHUNK_OVERLAP=100
RETRIEVAL_K=4
RETRIEVAL_CANDIDATES=10
FINAL_CONTEXT_K=4
BM25_WEIGHT=0.4
FAISS_WEIGHT=0.6

# ── Groq Cloud LLM Configuration ──
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1024
LLM_TOP_P=1.0

# ── Local Ollama Configuration ──
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# ── Ingestion Limits ──
MAX_UPLOAD_SIZE_BYTES=104857600  # 100 MB
TABULAR_ROWS_PER_CHUNK=200

# ── Feature Flags ──
QUERY_REWRITING_ENABLED=false
SHOW_THINKING_PROCESS=false
```

### Frontend (`merged/frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python**: Version 3.10 or 3.11
- **Node.js**: Version 18+ or 20+
- **Ollama** (Optional, for offline/local LLM): [Download Ollama](https://ollama.ai)

---

### Backend Setup

1. Open a terminal and navigate to the backend directory:
   ```bash
   cd merged/backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and supply your JWT_SECRET_KEY and GROQ_API_KEY
   ```
5. Start the backend API server:
   ```bash
   python api.py
   ```
   The backend will be available at `http://localhost:8000` (Interactive API docs at `http://localhost:8000/docs`).

---

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd merged/frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Verify or create `.env.local`:
   ```bash
   echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
   ```
4. Start the Next.js development server:
   ```bash
   npm run dev
   ```
5. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

---

### Docker & Docker Compose

To spin up the entire stack using Docker Compose:

```bash
cd merged/backend
docker compose up --build -d
```

This starts:
- Backend API on port `8000`
- Next.js Frontend on port `3000`
- Persisted volumes for vector indices and chat histories in `backend/vectorstore/`

---

## 📡 API Reference & Endpoints

All protected endpoints require the HTTP header:
`Authorization: Bearer <token>`

### Authentication Endpoints
- `POST /auth/register` — Create a new user account and return JWT.
  ```json
  { "username": "alice", "password": "securepassword123", "email": "alice@example.com" }
  ```
- `POST /auth/login` — Authenticate existing user and return JWT token.
  ```json
  { "username": "alice", "password": "securepassword123" }
  ```
- `GET /auth/me` — Return profile of the authenticated user.

### Documents & Upload Endpoints
- `GET /documents` — List all uploaded documents, total pages, chunks, and storage metrics for current user.
- `POST /upload` — Upload a document (`multipart/form-data`) to extract text, chunk, embed, and index into the user's FAISS vector store.
- `DELETE /documents/{filename}` — Remove a document from the index and recalculate index embeddings.
- `GET /status` — Return user's document index status and LLM availability.

### Chat & Streaming Endpoints
- `POST /stream` — Server-Sent Events (SSE) token stream for real-time answers.
  ```json
  {
    "question": "What are the main findings in section 3?",
    "conversation_id": "optional-uuid",
    "llm_mode": "auto",
    "use_query_rewriting": false,
    "show_thinking": false
  }
  ```
- `POST /chat` — Synchronous chat endpoint returning final answer, source citations, reasoning, and latency.

### Conversation Management Endpoints
- `GET /conversations` — List all chat conversations for the current user.
- `POST /conversations` — Create a new empty conversation session.
- `GET /conversations/{id}` — Fetch all historical messages and citations for a conversation.
- `PATCH /conversations/{id}` — Rename a conversation title.
- `DELETE /conversations/{id}` — Delete a conversation session and all its messages.

### Benchmarking Endpoints
- `GET /benchmark/models` — List all local Ollama models installed on the host machine.
- `POST /benchmark/run` — Run side-by-side benchmark comparing multiple models on test questions.
  ```json
  {
    "models": ["llama3.1:8b", "mistral:latest"],
    "questions": ["Summarize document key points", "List quantitative metrics"]
  }
  ```
- `GET /benchmark/results` — Fetch past benchmark execution results and scorecards.

### Analytics & System Health Endpoints
- `GET /health` — Public endpoint returning system status, registered user count, and active sessions.
- `GET /analytics/summary` — Returns user-specific telemetry: total queries, average latency, token estimates, and provider breakdown.

---

## 🎨 Frontend Application Features

### Authentication & Session Guard
- Secure login and registration with validation feedback.
- Client-side token validation on load with automatic redirect to `/auth`.

### Document Management & Library
- Drag-and-drop file upload with live progress indicators.
- Document library view showing file size, chunk counts, page counts, and deletion options.

### Chat Interface & Voice Interaction
- Modern chat UI with Markdown formatting, syntax-highlighted code blocks, and copy-to-clipboard.
- Expandable citation panel detailing document source name, page number, and similarity score.
- **Microphone Button**: Voice dictation using browser Web Speech API.
- **Speaker Button**: Neural text-to-speech audio read-out of generated answers.

### Offline / Online Provider Switch
- Header toggle for switching between **Auto**, **Online (Groq)**, and **Offline (Ollama)** modes.
- Instant connection status indicator showing whether Groq or Ollama is online.

### Model Benchmarking UI
- Interactive dashboard to select installed local models and execute automated comparative runs.
- Visual metrics for tokens per second, first-token latency, total latency, and output length.

### Analytics Dashboard
- Charts and metrics for query frequency, latency distribution, provider utilization, and vectorstore index density.

---

## 🧪 Local Model Benchmarking Suite

The built-in benchmark module (`src/benchmark.py`) enables objective evaluation of local LLMs:

1. **Retrieval Grounding**: Measures whether local models adhere strictly to retrieved context without hallucination.
2. **Speed & Throughput**: Tracks exact time-to-first-token (TTFT) and throughput in tokens/second.
3. **Execution**: Accessible directly from the `/benchmark` web UI or via `POST /benchmark/run`.

---

## 🔧 Troubleshooting & FAQ

### 1. `ModuleNotFoundError: No module named 'fastapi'`
Make sure your virtual environment is activated before starting the server:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Scanned PDFs return minimal or no text
Ensure Tesseract OCR is installed and available in your system `PATH`. PyMuPDF will automatically invoke Tesseract OCR when scanned non-text pages are detected.

### 3. Model Mismatch or Vector Dimension Errors
If you change `EMBEDDING_MODEL` in `.env`, you must rebuild your vector indices:
```bash
python rebuild_index.py --all
```

### 4. Ollama shows "Provider Offline"
Verify Ollama is running locally:
```bash
ollama list
ollama run llama3.1:8b
```
Ensure `OLLAMA_HOST=http://localhost:11434` is accessible.

### 5. Port 8000 or 3000 already in use
You can specify custom ports:
- **Backend**: `uvicorn api:app --port 8080` (update `NEXT_PUBLIC_API_URL` in frontend `.env.local`)
- **Frontend**: `npm run dev -- -p 3001`

---

## 📜 License

This project is licensed under the **MIT License**. You are free to use, modify, distribute, and integrate this software into commercial and open-source applications.
