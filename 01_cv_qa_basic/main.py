#!/usr/bin/env python3
"""
01_cv_qa_basic — Simple CV/Resume Q&A Chatbot

Reads a PDF resume, injects its text into a system prompt, and starts
an interactive loop where the user can ask questions via the OpenAI API.

Usage:
    uv run python 01_cv_qa_basic/main.py              # uses ../my_cv.pdf
    uv run python 01_cv_qa_basic/main.py path/to.pdf   # custom path
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# ── Configuration ────────────────────────────────────────────────────────────

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"

if not OPENAI_API_KEY:
    raise SystemExit("ERROR: OPENAI_API_KEY not found. Copy .env.example → .env and add your key.")

# ── PDF Extraction ───────────────────────────────────────────────────────────

def read_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

# ── Main Loop ────────────────────────────────────────────────────────────────

def main(pdf_path: str) -> None:
    cv_text = read_pdf(pdf_path)
    if not cv_text:
        print(f"ERROR: No text extracted from '{pdf_path}'.")
        sys.exit(1)

    system_prompt = (
        "You are my assistant. You are responsible for replying to questions "
        "about my CV/Resume and my career.\n"
        f"Here is my CV:\n{cv_text}\n\n"
        "Use it to reply to the user's question."
    )

    client = OpenAI()

    print(f"CV loaded ({len(cv_text)} chars). Type 'exit' to quit.\n")

    while True:
        user_prompt = input("Your question > ").strip()
        if not user_prompt:
            continue
        if user_prompt.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        print(response.choices[0].message.content, "\n")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "my_cv.pdf")
    if not os.path.isfile(pdf):
        print(f"File not found: {pdf}")
        print("Usage: uv run python 01_cv_qa_basic/main.py [path/to/cv.pdf]")
        sys.exit(1)
    main(pdf)
