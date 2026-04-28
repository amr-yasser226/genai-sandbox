<div align="center">

# GenAI Sandbox

**A collection of Generative AI experiments — from simple chatbots to streaming pipelines and LLM-powered routing.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://platform.openai.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-6366F1)](https://openrouter.ai)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://docs.astral.sh/uv/)

</div>

---

## Overview

This repository is a structured exploration of Generative AI concepts, developed incrementally to master the OpenAI API ecosystem and related tools. Each experiment is isolated within its own directory and focuses on specific implementation techniques:

| # | Experiment | Technique | Provider |
|---|-----------|-----------|----------|
| 01 | [CV Q&A (Basic)](#01-cv-qa-basic) | PDF extraction → system prompt → chat loop | OpenAI |
| 02 | [CV Q&A (Streaming)](#02-cv-qa-streaming) | Chunking, streaming tokens, Rich rendering | OpenRouter |
| 03 | [LLM Flow Control](#03-llm-flow-control) | Intent parsing → JSON routing → function dispatch | OpenAI |

---

## Repository Structure

```
genai-sandbox/
├── .env.example                          # API key template
├── .gitignore
├── .python-version                       # Python 3.11
├── pyproject.toml                        # Dependencies (managed by uv)
├── uv.lock
│
├── 01_cv_qa_basic/
│   ├── README.md
│   ├── main.py                           # CLI chatbot
│   └── chat_with_cv.ipynb                # Notebook walkthrough
│
├── 02_cv_qa_streaming/
│   ├── README.md
│   └── cv_qa_stream.py                   # Streaming CLI with Rich
│
└── 03_llm_flow_control/
    ├── README.md
    └── llm_flow_control.ipynb            # Calculator router notebook
```

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/) — High-performance Python package manager
- API credentials from [OpenAI](https://platform.openai.com/api-keys) and/or [OpenRouter](https://openrouter.ai/keys)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/amr-yasser226/genai-sandbox.git
cd genai-sandbox

# 2. Synchronize environment and install dependencies
uv sync

# 3. Configure environment variables
cp .env.example .env
# Update .env with valid API keys

# 4. Place a target PDF (e.g., resume) in the project root
cp /path/to/your/resume.pdf my_cv.pdf
```

---

## Experiments

### 01 — CV Q&A (Basic)

A baseline chatbot that implements PDF text extraction and utilizes the OpenAI API (`gpt-4o-mini`) to answer queries based on the provided document context.

```bash
uv run python 01_cv_qa_basic/main.py my_cv.pdf
```

**Technical Stack:** `openai`, `pypdf`, `python-dotenv`

---

### 02 — CV Q&A (Streaming)

An advanced implementation featuring real-time token streaming via OpenRouter. This version includes document chunking for context management and high-fidelity terminal rendering.

```bash
uv run python 02_cv_qa_streaming/cv_qa_stream.py my_cv.pdf
```

**Technical Stack:** `openai`, `pdfplumber`, `rich`, `python-dotenv`

**Key Features:**
- Asynchronous token streaming with immediate UI feedback
- Automated synchronous fallback mechanisms
- Structured Markdown rendering for terminal output

---

### 03 — LLM Flow Control

A demonstration of utilizing Large Language Models as intent-based routers. The system parses natural language inputs, generates structured JSON responses, and programmatically dispatches requests to internal Python functions.

```bash
uv run jupyter lab 03_llm_flow_control/llm_flow_control.ipynb
```

**Technical Stack:** `openai`, `json`

**Research Areas:**
- Intent-driven routing architectures
- Comparison of symbol-based vs. dynamic dispatch methods
- Error handling in LLM-integrated control flows

---

## Environment Configuration

| Variable | Scope | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | 01, 03 | OpenAI Platform API key |
| `OPENROUTER_API_KEY` | 02 | OpenRouter API key |
| `OPENROUTER_API_BASE` | 02 | API base URL (default: `https://openrouter.ai/api/v1`) |
| `OPENROUTER_MODEL` | 02 | Target model (default: `deepseek/deepseek-chat-v3-0324:free`) |

---

## Technical Stack Overview

| Category | Tools |
|----------|-------|
| **Core Language** | Python 3.11 |
| **Package Management** | [uv](https://docs.astral.sh/uv/) |
| **Inference APIs** | OpenAI, OpenRouter |
| **Document Processing** | pypdf, pdfplumber |
| **Interface Design** | Rich (CLI) |
| **Development** | Jupyter Lab |

---

## License

This repository is maintained for educational and research purposes.
