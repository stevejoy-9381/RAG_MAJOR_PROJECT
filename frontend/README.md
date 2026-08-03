# 🧠 DocMind — RAG Document Q&A System

> Upload any PDF. Ask anything. Get answers grounded entirely in your document — no hallucinations, no guesswork.

<div align="center">

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-Vercel-000000?style=for-the-badge&logo=vercel)](https://rag-frontend-cps5ix718-stevejoy-9381s-projects.vercel.app)
[![Backend](https://img.shields.io/badge/🤗%20Backend-HuggingFace%20Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)
[![Frontend Repo](https://img.shields.io/badge/Frontend-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/stevejoy-9381/RAG-FRONTEND-frontend)
[![Backend Repo](https://img.shields.io/badge/Backend-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/stevejoy-9381/RAG-SYSTEM-backend)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat&logo=nextdotjs)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-0077B5?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Docker Setup](#docker-setup)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🎯 About the Project

**DocMind** is a production-ready **Retrieval-Augmented Generation (RAG)** application that turns your static PDF documents into an interactive, AI-powered knowledge base.

Instead of asking a generic LLM that might hallucinate, DocMind retrieves the most relevant chunks from *your specific documents* and feeds them as context to Qwen 3.6 27B — so every answer is grounded in what's actually written in your files, with exact page-level citations.

Built as a full-stack project demonstrating real-world AI engineering: vector search, hybrid retrieval, streaming LLM responses, JWT authentication, and per-user document isolation.

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| 🎨 Frontend (Vercel) | https://rag-frontend-cps5ix718-stevejoy-9381s-projects.vercel.app |
| 🔧 Backend (HuggingFace Spaces) | https://huggingface.co/spaces |

> **Note:** The HuggingFace Spaces free tier sleeps after 30 minutes of inactivity. The first request after a sleep takes ~30–60 seconds to wake up.

---

## ✨ Features

- 📄 **PDF Upload** — Upload any PDF; the system automatically chunks, embeds, and indexes it
- 🔍 **Hybrid Retrieval** — Combines BM25 (keyword) + FAISS (semantic) search for higher accuracy than either alone
- ⚡ **Real-time Streaming** — Answers stream token-by-token via Server-Sent Events (SSE) — no waiting for the full response
- 🔐 **JWT Authentication** — Secure register/login with bcrypt password hashing and JWT tokens
- 👤 **Per-User Isolation** — Every user has their own private FAISS index; documents are never shared across accounts
- 💬 **Conversation Memory** — Sliding window of last 4 exchanges keeps multi-turn context within the LLM token budget
- 📚 **Source Citations** — Every answer includes the exact filename, page number, and a text preview of the source chunk
- 🗑️ **Document Management** — Upload multiple PDFs, view library stats, delete documents
- 🌙 **Dark / Light Mode** — Full theme support in the frontend UI
- 🐳 **Docker Ready** — Multi-stage Dockerfile for optimised production builds

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** | REST API framework with async support |
| **LangChain** | RAG pipeline orchestration |
| **FAISS** | Facebook AI Similarity Search — vector store |
| **HuggingFace Transformers** | `all-MiniLM-L6-v2` sentence embeddings |
| **Groq API** | Ultra-fast Qwen 3.6 27B inference |
| **BM25Retriever** | Keyword-based retrieval (sparse) |
| **EnsembleRetriever** | Hybrid BM25 + FAISS weighted retrieval |
| **PyMuPDF** | PDF loading and text extraction |
| **python-jose** | JWT token creation and verification |
| **passlib + bcrypt** | Secure password hashing |
| **Python 3.11** | Runtime |

### Frontend
| Technology | Purpose |
|---|---|
| **Next.js 14** | React framework with App Router |
| **TypeScript** | Type safety across the entire frontend |
| **Tailwind CSS** | Utility-first styling |
| **Lucide React** | Icon library |
| **clsx** | Conditional class names |

### Infrastructure
| Service | Purpose |
|---|---|
| **Hugging Face Spaces** | Backend hosting (Docker) |
| **Vercel** | Frontend hosting (Next.js) |
| **Docker** | Containerisation for backend |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│                      Next.js 14 Frontend                         │
│           (Vercel) — Auth · Chat · Sidebar · SSE Stream          │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTPS + Bearer Token
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                                │
│                 (Hugging Face Spaces)                            │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │  Auth Layer │   │   Ingestion  │   │   Retrieval + LLM    │  │
│  │  JWT + bcrypt│  │  Pipeline    │   │   Pipeline           │  │
│  └─────────────┘   └──────┬───────┘   └──────────┬───────────┘  │
│                           │                       │              │
│                     ┌─────▼──────┐        ┌──────▼──────┐       │
│                     │PyMuPDF Load│        │  Hybrid     │       │
│                     │Chunk·Embed │        │  Retriever  │       │
│                     └─────┬──────┘        │  BM25+FAISS │       │
│                           │               └──────┬──────┘       │
│                    ┌──────▼──────┐               │              │
│                    │ Per-User    │        ┌──────▼──────┐       │
│                    │ vectorstore/│        │  Qwen 3.6   │       │
│                    │ {user_id}/  │        │  27B        │       │
│                    └─────────────┘        │  Streaming  │       │
│                                           └─────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow

```
PDF Upload
    │
    ▼
PyMuPDF → Load pages → Enrich metadata (filename, page, timestamp)
    │
    ▼
RecursiveCharacterTextSplitter → Chunks (800 chars, 100 overlap)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2) → 384-dim vectors
    │
    ▼
FAISS Index (per user) → Save to vectorstore/{user_id}/faiss_index/
    │
    ▼
User asks question
    │
    ▼
EnsembleRetriever → BM25 (0.4 weight) + FAISS (0.6 weight) → Top-4 chunks
    │
    ▼
LangChain RetrievalQA → Prompt Template + Context + Question
    │
    ▼
Groq API (Qwen 3.6 27B) → Streamed tokens via SSE → Frontend
```

---

## 📁 Project Structure

```
RAG-SYSTEM-backend/
├── api.py                    # FastAPI app — all endpoints
├── requirements.txt          # Python dependencies
├── Dockerfile                # HuggingFace Spaces deployment
├── .env.example              # Environment variable template
├── src/
│   ├── __init__.py
│   ├── auth.py               # JWT creation/verification + bcrypt
│   ├── user_store.py         # User registry (JSON-based)
│   ├── ingest.py             # PDF load → chunk → embed → FAISS
│   ├── retriever.py          # Hybrid BM25+FAISS EnsembleRetriever
│   ├── llm.py                # Groq LLM config + prompt template
│   ├── chat_memory.py        # In-memory session history (sliding window)
│   └── document_store.py     # Per-user document registry
└── vectorstore/              # Created at runtime (ephemeral on HF)
    ├── users.json            # User accounts
    └── {user_id}/
        ├── documents.json    # User's document metadata
        └── faiss_index/      # User's FAISS vector index
            ├── index.faiss
            └── index.pkl

RAG-FRONTEND-frontend/
├── Dockerfile                # Multi-stage Next.js build
├── next.config.js
├── tailwind.config.ts
├── tsconfig.json
├── package.json
├── app/
│   ├── layout.tsx
│   ├── page.tsx              # Root → redirects to /chat
│   ├── globals.css
│   ├── auth/
│   │   └── page.tsx          # Login / Register page
│   └── chat/
│       └── page.tsx          # Main chat interface
├── components/
│   ├── ChatMessage.tsx       # Message bubble + streaming cursor
│   ├── Sidebar.tsx           # Document library + upload
│   └── SourcePanel.tsx       # Collapsible source citations
├── lib/
│   ├── api.ts                # Typed API client (fetch + SSE stream)
│   └── auth.ts               # JWT localStorage management
└── types/
    └── index.ts              # Shared TypeScript interfaces
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Git

---

### Backend Setup

```bash
# 1. Clone the backend repository
git clone https://github.com/stevejoy-9381/RAG-SYSTEM-backend
cd RAG-SYSTEM-backend

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your GROQ_API_KEY and JWT_SECRET_KEY

# 5. Run the development server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# API is now running at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

### Frontend Setup

```bash
# 1. Clone the frontend repository
git clone https://github.com/stevejoy-9381/RAG-FRONTEND-frontend
cd RAG-FRONTEND-frontend

# 2. Install dependencies
npm install

# 3. Set up environment variables
cp .env.local.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000

# 4. Run the development server
npm run dev

# Frontend is now running at http://localhost:3000
```

---

### Docker Setup

```bash
# Build and run backend with Docker
cd RAG-SYSTEM-backend

docker build -t docmind-api .

docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_key_here \
  -e JWT_SECRET_KEY=your_secret_here \
  docmind-api

# API available at http://localhost:7860
```

---

## 🔐 Environment Variables

### Backend `.env`

```env
# Required
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
JWT_SECRET_KEY=your_long_random_secret_string_at_least_32_chars

# Optional (defaults shown)
GROQ_MODEL=qwen/qwen3.6-27b
LLM_TEMPERATURE=0.2
CHUNK_SIZE=800
CHUNK_OVERLAP=100
RETRIEVAL_K=4
BM25_WEIGHT=0.4
FAISS_WEIGHT=0.6
ACCESS_TOKEN_EXPIRE_HOURS=24
```

### Frontend `.env.local`

```env
# Local development
NEXT_PUBLIC_API_URL=http://localhost:8000

# Production
NEXT_PUBLIC_API_URL=https://YOUR_USERNAME-docmind-api.hf.space
```

---

## 📡 API Reference

### Auth Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/register` | Create new account |
| `POST` | `/auth/login` | Login and receive JWT |
| `GET` | `/auth/me` | Verify token / get current user |

### Document Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Get user's document library stats |
| `POST` | `/upload` | Upload and index a PDF |
| `GET` | `/documents` | List all user documents |
| `DELETE` | `/documents/{filename}` | Delete a document |

### Chat Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/stream` | Ask a question (SSE streaming response) |
| `DELETE` | `/sessions/{session_id}` | Clear conversation memory |

### Example: Stream a Question

```bash
curl -X POST https://YOUR_SPACE.hf.space/stream \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?", "session_id": "uuid-here"}'
```

**Streaming Response Format (SSE):**
```
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " main"}
data: {"type": "token", "content": " topic"}
data: {"type": "metadata", "sources": [...], "session_id": "uuid"}
data: [DONE]
```

---

## ⚙️ How It Works

### 1. Document Ingestion

When a PDF is uploaded:
1. **PyMuPDF** loads and extracts text page by page
2. Metadata is enriched: filename, page number, upload timestamp, total pages
3. **RecursiveCharacterTextSplitter** splits text into 800-char chunks with 100-char overlap
4. **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) converts each chunk to a 384-dimensional vector
5. Vectors are stored in a **per-user FAISS index** (`vectorstore/{user_id}/faiss_index/`)
6. Document metadata is registered in `vectorstore/{user_id}/documents.json`

### 2. Retrieval

When a question is asked:
1. **BM25Retriever** finds the top-4 chunks by keyword matching (sparse retrieval)
2. **FAISS** finds the top-4 chunks by semantic similarity (dense retrieval)
3. **EnsembleRetriever** merges results with weights: BM25 (0.4) + FAISS (0.6)
4. Final top-4 unique chunks are selected as context

### 3. Generation

1. The 4 retrieved chunks + conversation history + user question are assembled into a prompt
2. The prompt is sent to **Groq API** (Qwen 3.6 27B)
3. Response tokens stream back via **SSE** to the frontend in real time
4. Once complete, source citations (file, page, preview) are sent as metadata
5. The exchange is saved to in-memory session history (sliding window of last 4 turns)

---

## ☁️ Deployment

### Backend → Hugging Face Spaces

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/docmind-api
cd docmind-api

# Copy backend files
cp -r /path/to/RAG-SYSTEM-backend/src .
cp /path/to/RAG-SYSTEM-backend/api.py .
cp /path/to/RAG-SYSTEM-backend/requirements.txt .
cp /path/to/RAG-SYSTEM-backend/Dockerfile .

git add .
git commit -m "deploy"
git push
```

Add secrets in HF Space Settings:
- `GROQ_API_KEY`
- `JWT_SECRET_KEY`

### Frontend → Vercel

```bash
# Push frontend to GitHub, then connect to Vercel
# Set environment variable in Vercel dashboard:
NEXT_PUBLIC_API_URL=https://YOUR_USERNAME-docmind-api.hf.space
```

> ⚠️ **Storage Note:** The free HuggingFace Spaces tier uses ephemeral storage. Uploaded documents and user accounts are lost when the Space restarts. For production persistence, migrate to a database (Supabase, PostgreSQL) and cloud object storage (S3, R2).

---

## 📸 Screenshots

| Auth Page | Chat Interface |
|---|---|
| *(Login / Register)* | *(Document Q&A with streaming)* |

| Document Sidebar | Source Citations |
|---|---|
| *(PDF library management)* | *(Page-level source panel)* |

---

## 🗺️ Roadmap

- [x] PDF upload and FAISS indexing
- [x] Hybrid BM25 + FAISS retrieval
- [x] Real-time SSE streaming
- [x] JWT authentication
- [x] Per-user document isolation
- [x] Conversation memory
- [x] Source citations
- [x] Docker deployment
- [ ] Persistent storage with Supabase / PostgreSQL
- [ ] Support for DOCX, TXT, CSV files
- [ ] Multi-document cross-search
- [ ] Re-ranking with a cross-encoder model
- [ ] Admin dashboard for user management
- [ ] Rate limiting per user

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Steve** — Full Stack Developer & Aspiring ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-stevejoy--9381-181717?style=flat&logo=github)](https://github.com/stevejoy-9381)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com)

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

*Built with ❤️ using FastAPI, LangChain, FAISS, and Next.js*

</div>
