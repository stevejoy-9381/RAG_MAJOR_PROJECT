# RAG Document Q&A System

A multi-user RAG (Retrieval-Augmented Generation) system with JWT authentication,
hybrid FAISS+BM25 retrieval, per-user document isolation, and dual LLM provider
support (local Ollama + cloud Groq).

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.api.txt

# 2. Configure
cp .env.example .env
# Edit .env — set JWT_SECRET_KEY (required), GROQ_API_KEY (optional)

# 3. Run
python api.py
# → http://localhost:8000
# → Docs at http://localhost:8000/docs
```

## Supported Document Formats

PDF, DOCX, PPTX, TXT, and CSV files are supported. Scanned/image-only PDFs are
automatically OCR'd if Tesseract is installed.

### OCR Setup (Optional — for scanned PDFs)

Scanned PDFs with no embedded text layer require [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on the host:

| Platform | Install Command |
|----------|----------------|
| **Linux (Debian/Ubuntu)** | `sudo apt-get install tesseract-ocr` |
| **macOS** | `brew install tesseract` |
| **Windows** | Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |
| **Docker** | Already included in both Dockerfiles |

> **Note:** If Tesseract is not installed, scanned pages will have minimal/no text but
> the upload will still succeed — OCR is a best-effort enhancement, not a hard requirement.

## Multilingual Support & Embeddings

You can query in non-English languages (e.g., Hindi, Telugu, Spanish, French) against English or multi-language documents.

To enable multilingual vector retrieval, update `.env`:

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

> **IMPORTANT WARNING:** Changing `EMBEDDING_MODEL` changes the vector embedding space. Existing indices generated with the old model will be incompatible.
>
> To reset your vector indices after changing models, run:
> ```bash
> python rebuild_index.py --all
> ```
> Then re-upload your documents. The assistant will retrieve cross-lingual passages and respond in the language your question was asked in.

## LLM Providers

The system supports two LLM providers, switchable per-request:

| Mode | Provider | Requires | Best For |
|------|----------|----------|----------|
| `offline` | Ollama (local) | Ollama running | Privacy, no internet, free |
| `online` | Groq (cloud) | `GROQ_API_KEY` | Speed, no GPU needed |
| `auto` | Auto-detect | Either one | Default — tries Ollama first, falls back to Groq |

### Local LLM Setup (Ollama)

**1. Install Ollama**

Download and install from [ollama.com](https://ollama.com):

- **Windows**: Download the installer from the website
- **macOS**: `brew install ollama` or download from the website
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`

**2. Pull a model**

```bash
ollama pull llama3.1:8b
```

This downloads the Llama 3.1 8B model (~4.7 GB). Other compatible models:
- `llama3.1:8b` — Good balance of quality and speed (recommended)
- `llama3.2:3b` — Faster, lower quality, less RAM
- `mistral:7b` — Alternative 7B model

**3. Verify installation**

```bash
# Check Ollama is running
ollama list

# Expected output:
# NAME              ID           SIZE    MODIFIED
# llama3.1:8b       ...          4.7 GB  ...
```

**4. Configure (optional)**

In your `.env` file:
```env
# Default values — usually no changes needed
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

**5. Test**

```bash
# Start the API (Ollama will be auto-detected)
python api.py

# Check provider availability
curl http://localhost:8000/health
```

### Cloud LLM Setup (Groq)

1. Get a free API key from [console.groq.com](https://console.groq.com)
2. Add to `.env`: `GROQ_API_KEY=your_key_here`

### Per-Request Provider Selection

The `/chat` and `/stream` endpoints accept an `llm_mode` field:

```json
{
  "question": "What is the main topic?",
  "llm_mode": "auto"
}
```

Values: `"auto"` (default), `"online"` (force Groq), `"offline"` (force Ollama)

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | No | Create account, get JWT |
| POST | `/auth/login` | No | Login, get JWT |
| GET | `/auth/me` | Yes | Verify token |
| GET | `/health` | No | Health check |
| GET | `/status` | Yes | User stats + LLM provider availability |
| POST | `/upload` | Yes | Upload document (PDF, DOCX, PPTX, TXT, CSV) |
| GET | `/documents` | Yes | List user's documents |
| DELETE | `/documents/{filename}` | Yes | Remove a document |
| POST | `/chat` | Yes | Ask a question (non-streaming) |
| POST | `/stream` | Yes | Ask a question (SSE streaming) |
| GET | `/sessions/{id}` | Yes | Get session history |
| DELETE | `/sessions/{id}` | Yes | Clear session |

## Docker

```bash
docker-compose up --build
```

See `docker-compose.yml` for an optional Ollama service block (commented out).

## Environment Variables

See [.env.example](.env.example) for all configuration options.
