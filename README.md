# Simple RAG Pipeline (PDF → FAISS → Groq)

This project demonstrates a clean, end-to-end implementation of a
Retrieval-Augmented Generation (RAG) system.

It shows how to take an unstructured document (a PDF), transform it into
a searchable knowledge base, and use a modern LLM to generate accurate,
context-aware answers. The goal is to provide a practical, minimal
example of how RAG systems work in real-world applications.

This system works best with PDF files that: - contain straightforward,
selectable text\
- are not scanned images\
- do not rely heavily on complex formatting

A sample file (`Trading-Volatility-Colin-Bennet.pdf`) is included and
was used to test this system. However, you can use **any PDF** by simply
changing the file path in the script.

------------------------------------------------------------------------

## Quick Start

```bash
pip install -r requirements.txt
python download_punkt.py
python build_index.py
python ask_llm.py
```

------------------------------------------------------------------------

## Key concepts demonstrated

-   Text extraction and preprocessing\
-   Chunking with overlap for better context retention\
-   Embedding generation using sentence-transformers\
-   Vector search with FAISS\
-   Retrieval-Augmented Generation using an LLM\
-   Prompt design to reduce hallucinations

------------------------------------------------------------------------

## How it works

The pipeline has three steps:

1.  **Indexing**\
    Extract text from the PDF, split it into chunks, convert to
    embeddings, and store in FAISS.

2.  **Retrieval**\
    Search the most relevant chunks based on a user query.

3.  **Generation**\
    Send retrieved content to an LLM to generate a final answer.

------------------------------------------------------------------------

## Installation

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## One-time setup (NLTK)

Before running the indexing step, download the tokenizer:

``` python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
```

------------------------------------------------------------------------

## Step 1 --- Build the index

Run:

``` bash
python build_index.py
```

By default the script uses the bundled sample document
(`Trading-Volatility-Colin-Bennet.pdf`). To index a different PDF, pass
its path as an argument:

``` bash
python build_index.py path/to/your.pdf
```

or set the `RAG_PDF_PATH` environment variable:

``` bash
export RAG_PDF_PATH=path/to/your.pdf
python build_index.py
```

### Output

This step generates:

-   `faiss_index.bin` --- vector index\
-   `chunks.pkl` --- text chunks

------------------------------------------------------------------------

## Step 2 --- Test retrieval

Run:

``` bash
python test_retrieval.py
```

You can ask a question and see the most relevant chunks retrieved from
the document.

This step is useful to verify that your embeddings and search are
working correctly before using an LLM.

------------------------------------------------------------------------

## Step 3 --- Ask questions with an LLM

This step completes the RAG pipeline using Groq.

### API Key setup

Provide your Groq API key through the `GROQ_API_KEY` environment
variable (recommended — never commit keys to the repository):

``` bash
# Linux / macOS
export GROQ_API_KEY="your_groq_api_key_here"

# Windows (PowerShell)
$env:GROQ_API_KEY = "your_groq_api_key_here"
```

For backwards compatibility, the script also accepts the key in a
local file named `groq.txt` (ignored by `.gitignore`) if the
environment variable is not set. The key is only read at runtime — it
is never printed or logged.

### Run

``` bash
python ask_llm.py
```

Ask a question, and the system will: - retrieve relevant chunks\
- send them to the model\
- generate a grounded answer

------------------------------------------------------------------------

## Configuration

All environment-specific values are configurable and default to safe,
portable values:

| Variable           | Purpose                       | Default                            |
|--------------------|-------------------------------|------------------------------------|
| `GROQ_API_KEY`     | Groq API key                  | (falls back to `groq.txt`)         |
| `RAG_PDF_PATH`     | PDF to index                  | `Trading-Volatility-Colin-Bennet.pdf` |
| `RAG_INDEX_PATH`   | FAISS index file              | `faiss_index.bin`                  |
| `RAG_CHUNKS_PATH`  | Serialized chunks file        | `chunks.pkl`                       |
| `RAG_DEBUG`        | `1` restores full tracebacks  | sanitized one-line errors          |

Scripts fail with short, sanitized messages (no stack traces, local
paths, or credentials in the output) via the shared `safe_output.py`
helper. Set `RAG_DEBUG=1` when you need the full traceback while
developing.

------------------------------------------------------------------------

## Notes

-   The system uses `all-MiniLM-L6-v2` for embeddings\
-   Retrieval uses FAISS with cosine similarity (via normalized
    embeddings)\
-   The LLM is instructed to only use retrieved context and say "I don't
    know" if the answer is not found

------------------------------------------------------------------------

## Project structure

    .
    ├── build_index.py
    ├── test_retrieval.py
    ├── ask_llm.py
    ├── download_punkt.py
    ├── safe_output.py
    ├── requirements.txt
    ├── .gitignore
    ├── groq.txt              # optional, local only, git-ignored
    ├── faiss_index.bin       # generated
    ├── chunks.pkl            # generated
    └── Trading-Volatility-Colin-Bennet.pdf

------------------------------------------------------------------------

## Summary

This project provides a practical, minimal implementation of a RAG
system that can:

-   process a PDF\
-   search its content efficiently\
-   generate answers grounded in the document

It is designed to be simple, extensible, and easy to adapt to other
datasets or use cases.
