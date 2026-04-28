<div align="center">

# 🧪 GenAI Sandbox

**A collection of Generative AI experiments — from simple chatbots to streaming pipelines and LLM-powered routing.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)](https://platform.openai.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-6366F1)](https://openrouter.ai)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)](https://docs.astral.sh/uv/)

</div>

---

## 📖 Overview

This repository is a hands-on exploration of **Generative AI** concepts, built incrementally while learning the OpenAI API ecosystem. Each experiment lives in its own directory and focuses on a specific technique:

| # | Experiment | Technique | Provider |
|---|-----------|-----------|----------|
| 01 | [CV Q&A (Basic)](#01--cv-qa-basic) | PDF extraction → system prompt → chat loop | OpenAI |
| 02 | [CV Q&A (Streaming)](#02--cv-qa-streaming) | Chunking, streaming tokens, Rich rendering | OpenRouter |
| 03 | [LLM Flow Control](#03--llm-flow-control) | Intent parsing → JSON routing → function dispatch | OpenAI |

---

## 🏗️ Repository Structure

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

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package manager
- An API key from [OpenAI](https://platform.openai.com/api-keys) and/or [OpenRouter](https://openrouter.ai/keys)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/amr-yasser226/genai-sandbox.git
cd genai-sandbox

# 2. Create a virtual environment and install dependencies
uv sync

# 3. Configure your API keys
cp .env.example .env
# Edit .env and paste your real keys

# 4. Add your CV/resume PDF to the project root
cp /path/to/your/resume.pdf my_cv.pdf
```

---

## 🔬 Experiments

### 01 — CV Q&A (Basic)

A minimal chatbot that loads your CV into a system prompt and answers questions via the OpenAI API.

```bash
uv run python 01_cv_qa_basic/main.py my_cv.pdf
```

**Stack:** `openai` · `pypdf` · `python-dotenv`

---

### 02 — CV Q&A (Streaming)

An advanced version that chunks the CV for context management, streams tokens in real time, and renders the final answer as formatted Markdown using [Rich](https://github.com/Textualize/rich).

```bash
uv run python 02_cv_qa_streaming/cv_qa_stream.py my_cv.pdf
```

**Stack:** `openai` · `pdfplumber` · `rich` · `python-dotenv`

**Features:**
- Real-time token streaming with live output
- Automatic sync fallback on stream failure
- Beautiful Markdown rendering in the terminal

---

### 03 — LLM Flow Control

A Jupyter notebook demonstrating how to use an LLM as an **intent router** — the model parses natural-language math expressions, outputs structured JSON, and the code dispatches to the correct arithmetic function.

```bash
uv run jupyter lab 03_llm_flow_control/llm_flow_control.ipynb
```

**Stack:** `openai` · `json`

**Explores:**
- Symbol-based routing (`+`, `-`, `*`, `/`) with if/elif chains
- Dynamic dispatch via `globals()` (and why it fails)

---

## ⚙️ Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | 01, 03 | Your OpenAI API key |
| `OPENROUTER_API_KEY` | 02 | Your OpenRouter API key |
| `OPENROUTER_API_BASE` | 02 | API base URL (default: `https://openrouter.ai/api/v1`) |
| `OPENROUTER_MODEL` | 02 | Model to use (default: `deepseek/deepseek-chat-v3-0324:free`) |

> Copy `.env.example` to `.env` and fill in your keys. The `.env` file is gitignored.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **LLM APIs** | OpenAI, OpenRouter |
| **PDF Parsing** | pypdf, pdfplumber |
| **CLI Rendering** | Rich |
| **Notebooks** | Jupyter Lab |

---

## 📄 License

This project is for educational and personal exploration purposes.
