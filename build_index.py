import os
import sys
import pickle

import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk

from safe_output import fail, install

# Sanitize console output before anything else can print or raise
install()

# Configuration (overridable via environment variables / CLI, no machine-specific paths in code)
DEFAULT_PDF_PATH = "Trading-Volatility-Colin-Bennet.pdf"
INDEX_PATH = os.environ.get("RAG_INDEX_PATH", "faiss_index.bin")
CHUNKS_PATH = os.environ.get("RAG_CHUNKS_PATH", "chunks.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# -----------------------------
# 1. Load PDF
# -----------------------------
def load_pdf(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n".join(text)


# -----------------------------
# 2. Split into chunks
# -----------------------------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # if adding sentence keeps chunk under limit
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())

            # create overlap (keep last part of previous chunk)
            current_chunk = current_chunk[-overlap:] + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -----------------------------
# 3. Create embeddings
# -----------------------------
def create_embeddings(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    return embeddings


# -----------------------------
# 4. Store in FAISS
# -----------------------------
def build_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    return index


# -----------------------------
# 5. Save index + chunks
# -----------------------------
def save_index(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # PDF path: CLI argument > environment variable > bundled sample PDF
    pdf_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("RAG_PDF_PATH", DEFAULT_PDF_PATH)
    )

    print("📖 Reading PDF...")
    try:
        text = load_pdf(pdf_path)
    except Exception:
        fail(f"Could not read PDF: {pdf_path}")

    print("✂️ Splitting into chunks...")
    try:
        chunks = split_text(text)
    except Exception:
        fail("Could not split the text into chunks. Is the PDF a text-based document?")
    print(f"Total chunks: {len(chunks)}")

    print("🧠 Creating embeddings...")
    try:
        embeddings = create_embeddings(chunks)
    except Exception:
        fail(
            f"Could not create embeddings with model '{EMBEDDING_MODEL}'. "
            "Check your network connection and installed dependencies."
        )

    print("📦 Building FAISS index...")
    index = build_faiss(embeddings)

    print("💾 Saving index...")
    try:
        save_index(index, chunks)
    except Exception:
        fail("Could not save the index files. Check write permissions.")

    print(f"✅ Done! Index saved as {INDEX_PATH}")
