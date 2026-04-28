# 02 — CV Q&A (Streaming)

An advanced CLI chatbot that reads your CV/resume from a PDF and streams responses in real time using **OpenRouter**, with formatted **Markdown output** via [Rich](https://github.com/Textualize/rich).

## Features

- **PDF text extraction** using `pdfplumber` (more robust than `pypdf`)
- **Context chunking** — splits large CVs into manageable pieces
- **Streaming output** — see tokens appear in real time
- **Sync fallback** — gracefully falls back to non-streaming if the stream fails
- **Rich Markdown** — final answer is rendered as formatted Markdown in the terminal

## Usage

```bash
# From the project root
uv run python 02_cv_qa_streaming/cv_qa_stream.py my_cv.pdf
```

## Required Environment Variables

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-chat-v3-0324:free
```
