# Chapter 05 — Tool System

> **Goal:** Understand how to define tools for Claude, implement their Python functions, and build a tool registry that the agent loop can dispatch through dynamically.

---

## 5.1 What Are Tools?

Tools are **capabilities you grant to Claude** — Python functions that Claude can request to be called during its reasoning process.  You define:
1. The tool's **name** and **description** (in natural language, for Claude to understand)
2. The **JSON Schema** for the tool's input parameters
3. The **Python function** that executes the tool and returns a string

Claude never executes Python code directly — it only *asks* you to call the function, then you feed the result back.

```
┌─────────────┐  "Call list_local_documents()"  ┌─────────────────┐
│   Claude    │ ─────────────────────────────▶  │  Your Backend   │
│             │                                  │  (Python)       │
│             │ ◀─────────────────────────────  │                 │
└─────────────┘  "Available documents:\n- ..."  └─────────────────┘
```

---

## 5.2 Tool Function Implementation

**Rule: Tools NEVER raise exceptions.** They catch every error and return it as a string starting with `"Error: "`. This lets Claude read the error and self-correct (e.g., retry with a different filename).

```python
# app/services/tools.py (infrastructure layer)
import logging
from pathlib import Path
from typing import Any, Callable

from app.core.config import settings

logger = logging.getLogger(__name__)

# Resolved once at import time — absolute, guaranteed path
DOCS_DIR: Path = Path(settings.local_data_dir).resolve()
DOCS_DIR.mkdir(parents=True, exist_ok=True)
```

### Tool 1: `list_local_documents`

```python
def list_local_documents() -> str:
    """Return a newline-separated list of all files in the sandboxed folder."""
    try:
        files = sorted(f.name for f in DOCS_DIR.iterdir() if f.is_file())
        if not files:
            return "The local_data directory is empty — no documents are available."
        return "Available documents:\n" + "\n".join(f"- {name}" for name in files)
    except Exception as exc:
        logger.error("list_local_documents failed: %s", exc, exc_info=True)
        return f"Error: Could not read the documents directory — {exc}"
```

### Tool 2: `read_local_document`

```python
def read_local_document(file_name: str) -> str:
    """Return the full UTF-8 text content of *file_name*, or an error string."""

    # ── Security: prevent directory traversal ─────────────────────────────── #
    try:
        requested = (DOCS_DIR / file_name).resolve()
    except Exception:
        return f"Error: '{file_name}' is not a valid file name."

    if DOCS_DIR not in requested.parents and requested != DOCS_DIR:
        logger.warning("Directory traversal attempt blocked: %s", file_name)
        return (
            f"Error: Access denied for '{file_name}'. "
            f"Only files inside the '{DOCS_DIR.name}' folder may be read."
        )

    if not requested.exists():
        return (
            f"Error: File '{file_name}' was not found. "
            "Call list_local_documents first to see what files are available."
        )
    if not requested.is_file():
        return f"Error: '{file_name}' is a directory, not a file."

    try:
        content = requested.read_text(encoding="utf-8")
        logger.info("Read file '%s' (%d chars)", file_name, len(content))
        return content
    except UnicodeDecodeError:
        return (
            f"Error: '{file_name}' cannot be read as UTF-8 text. "
            "The MVP only supports plain-text files (.txt, .md, .csv)."
        )
    except Exception as exc:
        logger.error("read_local_document('%s') failed: %s", file_name, exc, exc_info=True)
        return f"Error: Unexpected error reading '{file_name}' — {exc}"
```

---

## 5.3 Tool Definitions (JSON Schema for Claude)

This is what you pass in the `tools=` parameter of `messages.create()`.  Claude uses the `description` to decide *when* to call a tool, and the `input_schema` to format its arguments correctly.

```python
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_local_documents",
        "description": (
            "Lists all available document file names in the local knowledge base "
            "(the local_data folder). "
            "Call this tool FIRST when the user asks what documents are available, "
            "or whenever you need to discover file names before reading them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},   # ← no parameters needed
            "required": [],
        },
    },
    {
        "name": "read_local_document",
        "description": (
            "Reads the complete text content of a specific document from the local "
            "knowledge base. Call this when you need the actual content of a file "
            "to answer a question or produce a summary. "
            "Always use list_local_documents first if you are unsure of the exact file name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": (
                        "The exact name of the file to read, including its extension "
                        "(e.g., 'report.txt', 'data.md', 'sales.csv')."
                    ),
                }
            },
            "required": ["file_name"],
        },
    },
]
```

### Writing Good Tool Descriptions

The quality of your tool descriptions directly affects how well Claude uses them:

| Bad description | Good description |
|---|---|
| "Lists files." | "Lists all available document file names in the local knowledge base. Call this tool FIRST when the user asks what documents are available, or whenever you need to discover file names before reading them." |
| "Reads a file." | "Reads the complete text content of a specific document. Always use list_local_documents first if you are unsure of the exact file name." |

**Include:**
- When Claude should choose this tool (versus alternatives)
- What the tool returns (format, content)
- Pre-conditions or sequencing hints ("call X first")

---

## 5.4 The Tool Registry

The registry maps tool names (strings) to Python callables.  This is the **Open/Closed** pattern: `AgentService` never knows about specific tools — it only calls `execute_tool(name, input)`.

```python
TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "list_local_documents": lambda _inp: list_local_documents(),
    "read_local_document": lambda inp: read_local_document(inp["file_name"]),
}


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Dispatch a tool call by name and return its string result."""
    handler = TOOL_REGISTRY.get(tool_name)
    if handler is None:
        logger.error("Unknown tool requested: %s", tool_name)
        return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY)}"
    return handler(tool_input)
```

---

## 5.5 Security: Path Traversal Prevention

The most critical security concern for file-reading tools is directory traversal.  An attacker (or a hallucinating LLM) might request `file_name = "../../etc/passwd"`.

**The fix: `Path.resolve()` + parent check**

```python
requested = (DOCS_DIR / file_name).resolve()

# After resolve(), "../../../etc/passwd" becomes "/etc/passwd"
# Check it's still inside our sandbox:
if DOCS_DIR not in requested.parents and requested != DOCS_DIR:
    return "Error: Access denied ..."
```

`Path.resolve()` resolves all `..` components to an absolute path.  The parent check then verifies that the absolute path is still inside `DOCS_DIR`.

**Why this works:**
- `(DOCS_DIR / "../../etc/passwd").resolve()` → `/etc/passwd`
- `/etc/passwd`.parents = `[/, /etc]` — `DOCS_DIR` (`/your/app/local_data`) is NOT in that list → blocked

---

## 5.6 Adding a New Tool (Step-by-Step)

To add a `search_document` tool that searches for a keyword in a file:

**Step 1: Implement the function**
```python
def search_document(file_name: str, keyword: str) -> str:
    """Search for keyword in file_name and return matching lines."""
    content_result = read_local_document(file_name)   # reuse existing tool
    if content_result.startswith("Error:"):
        return content_result                          # propagate error string
    matches = [
        f"Line {i+1}: {line.strip()}"
        for i, line in enumerate(content_result.splitlines())
        if keyword.lower() in line.lower()
    ]
    if not matches:
        return f"No lines containing '{keyword}' found in '{file_name}'."
    return f"Found {len(matches)} match(es):\n" + "\n".join(matches)
```

**Step 2: Add the JSON schema**
```python
TOOL_DEFINITIONS.append({
    "name": "search_document",
    "description": "Search for a keyword in a specific document and return matching lines.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_name": {"type": "string", "description": "File to search."},
            "keyword": {"type": "string", "description": "Text to search for (case-insensitive)."},
        },
        "required": ["file_name", "keyword"],
    },
})
```

**Step 3: Register it**
```python
TOOL_REGISTRY["search_document"] = lambda inp: search_document(inp["file_name"], inp["keyword"])
```

**That's it. `AgentService` automatically picks up the new tool — no other file changes.**

---

## Chapter Summary

| Concept | Key Takeaway |
|---|---|
| Tools never raise | Return `"Error: ..."` strings — let Claude self-correct |
| Descriptions matter | Detailed descriptions improve tool selection accuracy |
| JSON Schema | Tells Claude the exact argument format for each tool |
| Tool Registry | Maps name → callable; `execute_tool()` dispatches dynamically |
| Path traversal | Always use `Path.resolve()` + parent check for file tools |
| Adding tools | Touch only `tools.py` — the agent loop is never modified |

---

Next: [Chapter 06 → The Agentic Loop](./06_agentic_loop.md)
