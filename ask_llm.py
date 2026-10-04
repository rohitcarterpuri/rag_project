import os
import sys
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

from safe_output import fail, install

# Sanitize console output before anything else can print or raise
install()

# Configuration (override via environment variables; no secrets in code)
INDEX_PATH = os.environ.get("RAG_INDEX_PATH", "faiss_index.bin")
CHUNKS_PATH = os.environ.get("RAG_CHUNKS_PATH", "chunks.pkl")
KEY_FILE = "groq.txt"  # legacy, git-ignored fallback location for the API key
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"


def load_api_key():
    """Read the Groq API key from the environment first.

    Falls back to the local, git-ignored KEY_FILE for backwards
    compatibility. The key is never printed or logged.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        if os.path.isfile(KEY_FILE):
            with open(KEY_FILE, "r") as f:
                return f.read().strip()
    except OSError:
        return ""
    return ""


# -----------------------------
# 1. Load Groq API key
# -----------------------------
GROQ_API_KEY = load_api_key()

if not GROQ_API_KEY:
    fail(
        "No Groq API key found. Set the GROQ_API_KEY environment variable "
        f"(recommended) or add the key to a local, git-ignored '{KEY_FILE}' file."
    )

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# 2. Load FAISS index + chunks
# -----------------------------
try:
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
except Exception:
    fail("Could not load the search index. Run 'python build_index.py' first.")

print(f"Loaded {len(chunks)} chunks")

# -----------------------------
# 3. Load embedding model
# -----------------------------
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
except Exception:
    fail(
        f"Could not load the embedding model '{EMBEDDING_MODEL}'. "
        "Check your network connection and installed dependencies."
    )

# -----------------------------
# 4. Ask question
# -----------------------------
query = input("Ask a question: ").strip()

if not query:
    print("❌ Please enter a valid question.")
    sys.exit()

# -----------------------------
# 5. Retrieve relevant chunks
# -----------------------------
query_embedding = model.encode([query], normalize_embeddings=True)

k = 5
D, I = index.search(np.array(query_embedding, dtype=np.float32), k)

retrieved_chunks = [chunks[i] for i in I[0]]

context = "\n\n".join(retrieved_chunks)

# -----------------------------
# 6. Send to Groq
# -----------------------------
prompt = f"""
You are a helpful assistant answering questions based on a book.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Give a detailed, well-explained answer with examples if possible.
"""

try:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
except Exception:
    # Provider errors can echo request details; keep the output generic.
    fail(
        "The LLM request failed. Check your API key, network connection, "
        "and that the selected model is available."
    )

# -----------------------------
# 7. Output answer
# -----------------------------
answer = response.choices[0].message.content

print("\n🤖 Answer:\n")
print(answer)
