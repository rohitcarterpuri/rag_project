import os
import sys
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from safe_output import fail, install

# Sanitize console output before anything else can print or raise
install()

# Configuration (overridable via environment variables; no machine-specific paths)
INDEX_PATH = os.environ.get("RAG_INDEX_PATH", "faiss_index.bin")
CHUNKS_PATH = os.environ.get("RAG_CHUNKS_PATH", "chunks.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# -----------------------------
# 1. Load FAISS index + chunks
# -----------------------------
try:
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
except Exception:
    fail("Could not load the search index. Run 'python build_index.py' first.")

print(f"Loaded {len(chunks)} chunks")

# -----------------------------
# 2. Load embedding model
# -----------------------------
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
except Exception:
    fail(
        f"Could not load the embedding model '{EMBEDDING_MODEL}'. "
        "Check your network connection and installed dependencies."
    )

# -----------------------------
# 3. Ask a question
# -----------------------------
query = input("Ask a question: ").strip()

if not query:
    print("❌ Please enter a valid question.")
    sys.exit()

# Convert query to embedding
query_embedding = model.encode([query], normalize_embeddings=True)

# -----------------------------
# 4. Search in FAISS
# -----------------------------
k = 5  # number of results
D, I = index.search(np.array(query_embedding, dtype=np.float32), k)

# -----------------------------
# 5. Show results
# -----------------------------
print("\n🔎 Top results:\n")

for i, idx in enumerate(I[0]):
    if idx == -1:
        continue
    print(f"Score: {D[0][i]:.4f}")
    print(chunks[idx])
    print("-" * 50)
