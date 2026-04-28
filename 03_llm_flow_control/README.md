# 03 — LLM Flow Control (Calculator Router)

A demonstration of using an LLM as an **intent router**. The model parses natural-language math requests, extracts the operation and operands, and dispatches to the correct Python function.

## How It Works

1. Define four arithmetic functions: `add`, `subtraction`, `multiplication`, `devide`.
2. Prompt GPT-4o-mini with a system prompt that instructs it to output structured JSON: `{"route": "+", "x": 2, "y": 3}`.
3. Parse the JSON and route to the matching function.
4. Explores two routing strategies:
   - **Symbol-based routing** (`+`, `-`, `*`, `/`) with explicit if/elif.
   - **Name-based routing** via `globals()` lookup (demonstrates the pitfall when route keys don't match function names).

## Key Takeaway

The notebook intentionally shows the `globals()` approach **failing** with a `KeyError` — a useful lesson in why structured routing with validation is preferable to dynamic dispatch.

## Usage

```bash
# Open in Jupyter
uv run jupyter lab 03_llm_flow_control/llm_flow_control.ipynb
```

## Required Environment Variable

```
OPENAI_API_KEY=sk-proj-...
```
