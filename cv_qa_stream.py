#!/usr/bin/env python3
"""
cv_qa_stream.py

- Use: uv add python-dotenv pdfplumber openai rich
- Run: uv run python cv_qa_stream.py my_cv.pdf
- .env keys:
    OPENROUTER_API_KEY=sk-...
    OPENROUTER_API_BASE=https://openrouter.ai/api/v1
    OPENROUTER_MODEL=openai/gpt-4o-mini   # or another model available to your key
"""
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
import pdfplumber
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

# Load env
load_dotenv()

# Configuration (use these env names)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

if not OPENROUTER_API_KEY:
    raise SystemExit("ERROR: OPENROUTER_API_KEY not found in .env")

# Create client pointed at OpenRouter
client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_API_BASE.rstrip("/"))

console = Console()

# --- PDF extraction ---------------------------------------------------------
def read_pdf_text(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()

def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks

# --- Messages ---------------------------------------------------------------
def build_messages(cv_chunks: List[str], question: str) -> List[dict]:
    system = {
        "role": "system",
        "content": (
            "You are an expert assistant specialized in reading CV/resume content and answering "
            "questions about a candidate. When you answer, produce a clear Markdown-formatted response. "
            "Use headings, bullet lists, bold for important facts, and short examples when needed. "
            "Be concise, factual, and cite the CV where appropriate."
        ),
    }
    messages = [system]
    for idx, chunk in enumerate(cv_chunks, start=1):
        messages.append({"role": "user", "content": f"[CV PART {idx}]\n{chunk}"})
    messages.append({"role": "user", "content": f"Question about the CV:\n{question}\n\nPlease answer in Markdown."})
    return messages

# --- Helpers to extract text from streamed chunks ---------------------------
def _extract_text_from_stream_chunk(chunk) -> Optional[str]:
    """
    The streaming chunk can have several shapes (dict-like or object-like).
    Try common locations where text may appear and return the fragment or None.
    """
    # dict-like chunk (older style)
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            c0 = choices[0]
            delta = c0.get("delta") or {}
            content = None
            if isinstance(delta, dict):
                content = delta.get("content")
            # final message shape
            if not content and c0.get("message"):
                msg = c0.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
            if isinstance(content, str) and content:
                return content
        if "text" in chunk and isinstance(chunk["text"], str):
            return chunk["text"]
        return None

    # object-like chunk (new client event)
    try:
        # token (some event shapes)
        if hasattr(chunk, "token") and getattr(chunk, "token"):
            return getattr(chunk, "token")
        # delta object (may have .content)
        if hasattr(chunk, "delta") and getattr(chunk, "delta") is not None:
            delta = getattr(chunk, "delta")
            if isinstance(delta, dict):
                return delta.get("content")
            if hasattr(delta, "content"):
                return getattr(delta, "content")
        # data dict (raw)
        if hasattr(chunk, "data") and getattr(chunk, "data") is not None:
            data = getattr(chunk, "data")
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if isinstance(delta, dict):
                        return delta.get("content")
        # message attribute
        if hasattr(chunk, "message") and getattr(chunk, "message") is not None:
            msg = getattr(chunk, "message")
            if isinstance(msg, dict):
                return msg.get("content")
            if hasattr(msg, "content"):
                return getattr(msg, "content")
    except Exception:
        return None
    return None

# --- Streaming / fallback logic ---------------------------------------------
def stream_chat_completion(messages: List[dict], model: str = OPENROUTER_MODEL):
    """
    Stream tokens using create(..., stream=True) and yield token fragments.
    This implementation iterates the returned generator (widely supported pattern)
    and uses a robust extractor for different chunk shapes.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )
        any_yielded = False
        for chunk in resp:
            text = _extract_text_from_stream_chunk(chunk)
            if text:
                any_yielded = True
                yield text
        # if the generator returned but we didn't yield anything, caller may fallback
        if not any_yielded:
            return
    except Exception as e:
        raise RuntimeError(f"OpenAI/OpenRouter streaming error: {e}") from e

def fetch_sync_completion(messages: List[dict], model: str = OPENROUTER_MODEL) -> str:
    """Non-streaming fallback; returns final assistant text (if present)."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    # dict-like
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if choices:
            c0 = choices[0]
            if c0.get("message"):
                msg = c0["message"]
                if isinstance(msg, dict):
                    return msg.get("content") or ""
            if c0.get("text"):
                return c0.get("text") or ""
        return ""
    # object-like
    if hasattr(resp, "choices") and getattr(resp, "choices"):
        c0 = getattr(resp, "choices")[0]
        if hasattr(c0, "message") and getattr(c0, "message"):
            msg = getattr(c0, "message")
            if isinstance(msg, dict):
                return msg.get("content") or ""
            if hasattr(msg, "content"):
                return getattr(msg, "content") or ""
        if hasattr(c0, "text") and getattr(c0, "text"):
            return getattr(c0, "text")
    return ""

def list_available_models() -> List[str]:
    """Return list of model ids/names visible to the API key (useful to confirm model names)."""
    try:
        resp = client.models.list()
        models = []
        # resp may be dict-like or object-like; try common shapes
        if isinstance(resp, dict):
            for m in resp.get("data", []):
                models.append(m.get("id") or m.get("name") or str(m))
        else:
            if hasattr(resp, "data"):
                for m in getattr(resp, "data"):
                    if isinstance(m, dict):
                        models.append(m.get("id") or m.get("name") or str(m))
                    else:
                        if hasattr(m, "id"):
                            models.append(getattr(m, "id"))
                        elif hasattr(m, "name"):
                            models.append(getattr(m, "name"))
        return models
    except Exception as exc:
        return [f"Could not list models: {exc}"]

# --- Main interactive loop --------------------------------------------------
def main(pdf_path: str):
    console.print(f"Reading CV from: [bold]{pdf_path}[/bold]")
    cv_text = read_pdf_text(pdf_path)
    if not cv_text:
        console.print("[red]No text found in PDF.[/red]")
        return
    console.print(f"Extracted CV length: {len(cv_text)} characters")
    chunks = chunk_text(cv_text, max_chars=3000)
    console.print(f"Prepared [bold]{len(chunks)}[/bold] chunk(s) for context.")

    console.print("\nEnter questions about this CV. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = console.input("[cyan]Your question[/cyan] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]Exiting.[/bold]")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            console.print("[bold]Goodbye.[/bold]")
            break

        messages = build_messages(chunks[:3], question)
        console.rule("[bold]Streaming answer (raw tokens)[/bold]")
        buffer: List[str] = []
        try:
            for token in stream_chat_completion(messages):
                print(token, end="", flush=True)
                buffer.append(token)
        except RuntimeError as e:
            console.print(f"\n[red]Error during streaming: {e}[/red]")
            console.print("[yellow]Attempting synchronous fallback...[/yellow]")
            try:
                final = fetch_sync_completion(messages)
                if final:
                    print(final)
                    buffer.append(final)
            except Exception as e2:
                console.print(f"[red]Synchronous fallback failed: {e2}[/red]")
                continue

        # If streaming produced nothing, final fallback
        if not buffer:
            try:
                final = fetch_sync_completion(messages)
                if final:
                    print(final)
                    buffer.append(final)
            except Exception as e:
                console.print(f"[red]Final fetch failed: {e}[/red]")
                continue

        full_answer = "".join(buffer).strip()
        print()
        console.rule("[bold]Formatted Markdown output[/bold]")
        md = Markdown(full_answer or "*No output produced.*")
        console.print(md)
        console.rule()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python cv_qa_stream.py /path/to/cv.pdf")
        sys.exit(1)
    pdf_path_arg = sys.argv[1]
    if not os.path.isfile(pdf_path_arg):
        print(f"File not found: {pdf_path_arg}")
        sys.exit(1)
    main(pdf_path_arg)
