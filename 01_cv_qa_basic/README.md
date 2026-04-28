# 01 — CV Q&A (Basic)

A minimal chatbot that reads your CV/resume from a PDF and answers questions about it using the **OpenAI API** (`gpt-4o-mini`).

## How It Works

1. Extracts text from a PDF using `pypdf`.
2. Injects the full CV text into a system prompt.
3. Starts an interactive loop — you ask questions, the model answers.

## Files

| File | Description |
|------|-------------|
| `main.py` | CLI chatbot — run from the terminal |
| `chat_with_cv.ipynb` | Jupyter notebook version (step-by-step walkthrough) |

## Usage

```bash
# From the project root
uv run python 01_cv_qa_basic/main.py my_cv.pdf
```

## Required Environment Variable

```
OPENAI_API_KEY=sk-proj-...
```
