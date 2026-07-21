"""
src/ingest.py — Per-User Ingestion Pipeline (Phase 6: Multi-Format)
────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM PHASE 3:

  Phase 3: Only accepted .pdf files via PyMuPDFLoader.
  Phase 6: Accepts .pdf, .docx, .pptx, .txt, .csv — each with a
           dedicated loader that normalizes output into the same
           LangChain Document shape (page_content + metadata).

  Everything downstream of loading (enrich_metadata, chunk_documents,
  append_to_user_index) is UNCHANGED — it operates on generic Document
  lists regardless of source format.

SUPPORTED FORMATS:
  .pdf  → PyMuPDFLoader (one Document per page)
  .docx → docx2txt via Docx2txtLoader (one Document, page=0)
  .pptx → python-pptx (one Document per slide, page=slide number)
  .txt  → plain read (one Document, page=0)
  .csv  → CSVLoader (one Document per row)
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.document_store import register_document
from src.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".csv"}


def get_user_index_path(user_id: str) -> str:
    """Return the FAISS index path for a specific user."""
    return f"vectorstore/{user_id}/faiss_index"


# ─── Per-format loaders ──────────────────────────────────────────────────────
# Each returns a list of LangChain Document objects with .page_content and
# .metadata["page"]. This is the same shape PyMuPDFLoader produces, so
# enrich_metadata() and chunk_documents() work unchanged downstream.

# Minimum character threshold for a page to be considered "has real text".
# Pages below this threshold are likely scanned/image-only and get OCR'd.
OCR_TEXT_THRESHOLD = int(os.getenv("OCR_TEXT_THRESHOLD", 20))


def _ocr_page(file_path: str, page_num: int) -> str | None:
    """
    Render a single PDF page to image and OCR it with Tesseract.

    Uses PyMuPDF's page.get_pixmap() to render at 300 DPI (good OCR quality
    without excessive memory), then pytesseract for text extraction.

    Returns the OCR'd text string, or None if Tesseract is not installed
    or OCR fails for any reason. Errors are logged but never crash the pipeline.
    """
    try:
        import pytesseract
        from PIL import Image
        import fitz  # PyMuPDF
    except ImportError as e:
        print(f"[INGEST] ⚠ OCR dependencies missing ({e}). Skipping OCR.")
        return None

    try:
        pdf_doc = fitz.open(file_path)
        page = pdf_doc[page_num]

        # Render at 300 DPI (default is 72 DPI — too low for OCR)
        # zoom = 300/72 ≈ 4.17x
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to PIL Image for pytesseract
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(img)
        pdf_doc.close()
        return text.strip()

    except Exception as e:
        # Catch Tesseract not installed, corrupt pages, memory issues, etc.
        error_msg = str(e)
        if "tesseract" in error_msg.lower() or "not installed" in error_msg.lower():
            print(f"[INGEST] ⚠ Tesseract not installed. Install with:")
            print(f"[INGEST]   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print(f"[INGEST]   Linux:   apt-get install tesseract-ocr")
            print(f"[INGEST]   macOS:   brew install tesseract")
        else:
            print(f"[INGEST] ⚠ OCR failed for page {page_num + 1}: {e}")
        return None


def _load_pdf(file_path: str) -> list:
    """
    Load PDF using PyMuPDFLoader with OCR fallback for scanned pages.

    Two-pass approach:
      1. PyMuPDFLoader extracts embedded text (fast, reliable for text PDFs)
      2. For any page where the extracted text is too short (<OCR_TEXT_THRESHOLD
         chars), render that page to an image and OCR it with Tesseract

    OCR'd pages get metadata["ocr"] = True so they're distinguishable downstream
    (e.g. for confidence indicators in the frontend).

    If Tesseract isn't installed, scanned pages are kept with their minimal text
    and a warning is logged — the upload does NOT fail.
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    ocr_count = 0
    tesseract_available = True  # flip to False on first failure

    for i, doc in enumerate(docs):
        text_len = len(doc.page_content.strip())
        if text_len < OCR_TEXT_THRESHOLD and tesseract_available:
            print(f"[INGEST] Page {i + 1}: only {text_len} chars — attempting OCR...")
            ocr_text = _ocr_page(file_path, i)

            if ocr_text is None:
                # Tesseract not available — stop trying for remaining pages
                tesseract_available = False
                continue

            if len(ocr_text) > text_len:
                doc.page_content = ocr_text
                doc.metadata["ocr"] = True
                ocr_count += 1
                print(f"[INGEST] Page {i + 1}: OCR recovered {len(ocr_text)} chars")
            else:
                print(f"[INGEST] Page {i + 1}: OCR produced no improvement ({len(ocr_text)} chars)")

    if ocr_count > 0:
        print(f"[INGEST] Loaded {len(docs)} pages (PDF, {ocr_count} OCR'd)")
    else:
        print(f"[INGEST] Loaded {len(docs)} pages (PDF)")
    return docs


def _load_docx(file_path: str) -> list:
    """
    Load .docx using Docx2txtLoader — one Document for the whole file.

    Docx2txtLoader extracts all text from the document (including tables,
    headers, footers) into a single Document. We set page=0 since DOCX
    files don't have a meaningful page concept without rendering.
    """
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    # Ensure page metadata exists (Docx2txtLoader doesn't set it)
    for doc in docs:
        doc.metadata.setdefault("page", 0)
    print(f"[INGEST] Loaded 1 document section (DOCX, {len(docs[0].page_content)} chars)")
    return docs


def _load_pptx(file_path: str) -> list:
    """
    Load .pptx using python-pptx — one Document per slide.

    Extracts all text frames from each slide and joins them with newlines.
    Sets page metadata to the slide number (0-indexed, matching PDF convention).
    Skips empty slides (no text content).
    """
    from pptx import Presentation
    prs = Presentation(file_path)
    docs = []
    for slide_idx, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        texts.append(text)
        content = "\n".join(texts)
        if content:  # skip empty slides
            docs.append(Document(
                page_content=content,
                metadata={"page": slide_idx},
            ))
    print(f"[INGEST] Loaded {len(docs)} slides (PPTX)")
    return docs


def _load_txt(file_path: str) -> list:
    """
    Load plain text file — one Document, page=0.

    Uses UTF-8 with fallback to latin-1 for files with non-UTF-8 encoding.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()
    docs = [Document(page_content=content, metadata={"page": 0})]
    print(f"[INGEST] Loaded 1 document (TXT, {len(content)} chars)")
    return docs


def _load_csv(file_path: str) -> list:
    """
    Load CSV using CSVLoader — one Document per row.

    Each row becomes a Document with the cell values formatted as
    "column: value" pairs. Sets page=0 for all rows (CSVs don't have pages).
    """
    from langchain_community.document_loaders import CSVLoader
    loader = CSVLoader(file_path, encoding="utf-8")
    try:
        docs = loader.load()
    except UnicodeDecodeError:
        loader = CSVLoader(file_path, encoding="latin-1")
        docs = loader.load()
    # Ensure page metadata exists (CSVLoader doesn't set it)
    for doc in docs:
        doc.metadata.setdefault("page", 0)
    print(f"[INGEST] Loaded {len(docs)} rows (CSV)")
    return docs


# ─── Dispatcher ───────────────────────────────────────────────────────────────

_LOADERS = {
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".pptx": _load_pptx,
    ".txt":  _load_txt,
    ".csv":  _load_csv,
}


def load_document(file_path: str) -> list:
    """
    Load a document file, dispatching to the appropriate format-specific loader.

    Supported formats: .pdf, .docx, .pptx, .txt, .csv
    All loaders return the same Document shape so downstream processing
    (enrich_metadata, chunk_documents) works unchanged.

    Raises:
        FileNotFoundError: if the file doesn't exist
        ValueError: if the file extension is not supported
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    loader_fn = _LOADERS.get(ext)
    if loader_fn is None:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {supported}")

    return loader_fn(str(path))


def enrich_metadata(documents: list, original_filename: str) -> list:
    """Attach clean metadata to each page (filename, time, total pages)."""
    upload_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    total_pages = len(documents)
    for doc in documents:
        doc.metadata["source"]      = original_filename
        doc.metadata["upload_time"] = upload_time
        doc.metadata["total_pages"] = total_pages
    return documents


def chunk_documents(documents: list) -> list:
    """Split pages into well-sized, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ". ", "; ", ": ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Add chunk index metadata for richer citations
    source_counts: dict = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        chunk.metadata["chunk_index"] = source_counts[src]

    avg = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    print(f"[INGEST] {len(chunks)} chunks created (avg {avg} chars)")
    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and return the embedding model (downloads once, cached locally)."""
    print(f"[INGEST] Loading embedding model...")
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[INGEST] Embedding model ready.")
    return model


def append_to_user_index(
    user_id: str,
    chunks: list,
    embedding_model: HuggingFaceEmbeddings,
) -> int:
    """
    Append new document chunks to this user's FAISS index.

    Identical logic to Phase 1 append_to_index() but uses the
    per-user path from get_user_index_path().
    """
    index_path = get_user_index_path(user_id)
    os.makedirs(index_path, exist_ok=True)

    print(f"[INGEST] Embedding {len(chunks)} chunks for user '{user_id[:8]}'...")
    new_vs = FAISS.from_documents(chunks, embedding_model)

    index_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_file):
        print("[INGEST] Existing index found — merging...")
        existing_vs = FAISS.load_local(
            index_path, embedding_model,
            allow_dangerous_deserialization=True,
        )
        existing_vs.merge_from(new_vs)
        existing_vs.save_local(index_path)
        print("[INGEST] Merge complete.")
    else:
        new_vs.save_local(index_path)
        print("[INGEST] Fresh index created.")

    return len(chunks)


def run_ingestion(
    file_path: str,
    user_id: str,
    original_filename: str = None,
) -> dict:
    """
    Master ingestion function — now requires user_id for isolation.

    PIPELINE:
      file → load → enrich metadata → chunk → embed
            → append to user's FAISS index → register in user's document store
    """
    display_name = original_filename or Path(file_path).name
    file_size_kb = Path(file_path).stat().st_size / 1024

    print(f"\n[INGEST] user='{user_id[:8]}' file='{display_name}'")

    documents      = load_document(file_path)
    documents      = enrich_metadata(documents, display_name)
    chunks         = chunk_documents(documents)
    embedding_model = get_embedding_model()
    total_chunks   = append_to_user_index(user_id, chunks, embedding_model)

    register_document(
        user_id=user_id,
        filename=display_name,
        pages=len(documents),
        chunks=total_chunks,
        size_kb=file_size_kb,
    )

    print(f"[INGEST] [OK] Done: {len(documents)} pages, {total_chunks} chunks\n")
    return {
        "status": "success",
        "file":   display_name,
        "pages":  len(documents),
        "chunks": total_chunks,
    }
