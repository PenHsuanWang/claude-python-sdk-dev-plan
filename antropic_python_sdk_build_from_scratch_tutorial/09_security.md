# Chapter 09 — Security Patterns

> **Goal:** Understand the three security patterns woven throughout this codebase: path traversal prevention, API key protection, and graceful error degradation.

---

## 9.1 The Threat Model

This agent serves requests that can trigger **file system access**.  The two primary attack vectors are:

1. **Directory Traversal** — a client (or a hallucinating LLM) requests `../../etc/passwd`
2. **Secret Exposure** — the API key appears in logs, error messages, or HTTP responses
3. **Crash-on-Error** — unhandled exceptions expose stack traces or crash the agent loop

Each is addressed by a specific pattern in the code.

---

## 9.2 Path Traversal Prevention

### The Attack

```bash
# Attacker tries to read outside the sandbox
curl -X POST http://localhost:8000/api/v1/chat \
  -d '{"session_id": "x", "user_message": "Read the file ../../etc/passwd"}'

# Or via the upload API
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@malicious.txt;filename=../../evil.txt"
```

### Layer 1: Filename Sanitization (Documents API)

```python
# app/api/v1/documents.py
from pathlib import PurePosixPath
import re

_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-. ]*$")

def _validate_filename(filename: str) -> str:
    # Step 1: strip ALL path components
    safe_name = PurePosixPath(filename).name
    # "../../evil.txt" → "evil.txt" (PurePosixPath.name = last component only)

    # Step 2: allowlist regex — no slashes, no null bytes, no special chars
    if not _SAFE_FILENAME_RE.match(safe_name):
        raise HTTPException(status_code=400, detail="Invalid filename characters.")

    # Step 3: allowlist extensions
    ext = Path(safe_name).suffix.lower()
    if ext not in {".txt", ".md", ".csv"}:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    return safe_name
```

### Layer 2: `Path.resolve()` + Parent Check (Both Tools and API)

```python
# app/services/tools.py  AND  app/api/v1/documents.py

DOCS_DIR: Path = Path(settings.local_data_dir).resolve()  # absolute path

# When accessing any file:
requested = (DOCS_DIR / file_name).resolve()
# resolve() expands all ".." → "/etc/passwd" etc.

# The check:
if DOCS_DIR not in requested.parents and requested != DOCS_DIR:
    # The resolved path is NOT inside our sandbox → BLOCKED
    return "Error: Access denied ..."
```

**Why `resolve()` is the critical step:**

```python
from pathlib import Path

DOCS_DIR = Path("/app/local_data")

# Attacker input
evil = (DOCS_DIR / "../../etc/passwd").resolve()
# → Path("/etc/passwd")

# Parent check
print(DOCS_DIR in evil.parents)  # False → BLOCKED ✓

# Normal input  
normal = (DOCS_DIR / "report.txt").resolve()
# → Path("/app/local_data/report.txt")

print(DOCS_DIR in normal.parents)  # True → ALLOWED ✓
```

### Defense in Depth

The same `Path.resolve()` check exists in **both** the API layer (`documents.py`) and the tool layer (`tools.py`).  Even if a client bypasses the API (by directly crafting a message to Claude that asks it to use a tool with a traversal path), the tool layer blocks it independently.

---

## 9.3 API Key Protection with `SecretStr`

`pydantic`'s `SecretStr` wraps a string to prevent it from appearing in:
- `repr()` / `print()` output
- JSON serialization
- Log messages that include the settings object

```python
# app/core/config.py
from pydantic import SecretStr

class Settings(BaseSettings):
    anthropic_api_key: SecretStr = Field(..., alias="ANTHROPIC_API_KEY")
```

**Demonstration:**

```python
>>> from app.core.config import settings
>>> print(settings)
# anthropic_api_key=SecretStr('**********')   ← hidden ✓

>>> print(settings.anthropic_api_key)
# **********    ← hidden ✓

>>> import json
>>> json.dumps({"key": settings.anthropic_api_key})
# TypeError: Object of type SecretStr is not JSON serializable  ← can't leak ✓

>>> settings.anthropic_api_key.get_secret_value()
# "sk-ant-your-actual-key"   ← only visible when explicitly requested
```

**The only place `.get_secret_value()` is called:**

```python
# app/services/agent.py — the one legitimate use
_client = AsyncAnthropic(
    api_key=settings.anthropic_api_key.get_secret_value(),
    ...
)
```

By design, this is the only `.get_secret_value()` call in the entire codebase.  Any other occurrence should be treated as a security issue.

---

## 9.4 Graceful Error Degradation in Tools

Tools are the boundary between your Python code and Claude's reasoning.  Exceptions in tools crash the agent loop — but returning error strings lets Claude self-correct.

**The pattern:**

```python
def read_local_document(file_name: str) -> str:
    try:
        content = requested.read_text(encoding="utf-8")
        return content
    except FileNotFoundError:
        # Return error string — Claude reads this and can retry with a correct filename
        return f"Error: File '{file_name}' was not found. Call list_local_documents first."
    except UnicodeDecodeError:
        return f"Error: '{file_name}' cannot be read as UTF-8 text."
    except Exception as exc:
        # Catch-all — still no exception propagation
        logger.error("read_local_document failed: %s", exc, exc_info=True)
        return f"Error: Unexpected error reading '{file_name}' — {exc}"
```

**What happens when Claude receives an error string:**

```
User: "Read report.Txt"
Claude: [tool_use: read_local_document("report.Txt")]
Tool:   "Error: File 'report.Txt' was not found. Call list_local_documents first."
Claude: [tool_use: list_local_documents()]
Tool:   "Available documents:\n- report.txt"
Claude: [tool_use: read_local_document("report.txt")]   ← self-corrected the filename
Tool:   "Q3 Revenue: $4.2M ..."
Claude: "Here is a summary of report.txt: ..."
```

Claude acts as its own error handler — it reads the error, understands the problem, and retries with corrected parameters.  This only works if **tools never raise exceptions**.

---

## 9.5 Error Handling in the Presentation Layer

The HTTP layer adds another safety net — typed domain errors are mapped to safe user messages, and unknown exceptions are logged but never surfaced:

```python
# app/api/v1/chat.py
try:
    answer = await agent_service.run(session)
    return ChatResponse(response=answer, status="success")

except AgentError as exc:
    # Known domain error → safe, generic user message
    logger.warning("Agent error: %s", exc)
    return ChatResponse(
        response="The agent encountered an error and could not complete your request.",
        status="error",
    )
except Exception as exc:
    # Unknown error → log with full traceback (for developers), generic response for user
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return ChatResponse(
        response="An unexpected internal error occurred. Please try again.",
        status="error",
    )
```

**What never reaches the client:**
- Python stack traces
- File paths on the server
- Internal variable names or values
- The Anthropic API key or any secret

---

## 9.6 Security Checklist

Use this when extending the agent:

```
✅ Path traversal
   [ ] Use Path.resolve() for every file access
   [ ] Check DOCS_DIR in resolved_path.parents after resolve()
   [ ] Validate extension against allowlist
   [ ] Strip path components from user-provided filenames

✅ Secrets
   [ ] API key is SecretStr — never str
   [ ] .get_secret_value() called ONLY when passing to SDK
   [ ] .env in .gitignore
   [ ] No secrets in log messages

✅ Error handling
   [ ] Tools return "Error: ..." strings — never raise
   [ ] AgentError caught in handler → user-safe message
   [ ] Exception caught in handler → log + generic message
   [ ] No internal paths, types, or traces in HTTP responses

✅ Input validation
   [ ] Filenames validated with regex allowlist
   [ ] File content validated as UTF-8 before writing
   [ ] user_message bounded by min_length/max_length in Pydantic schema
```

---

## Chapter Summary

| Pattern | Implementation | What it prevents |
|---------|---------------|-----------------|
| `Path.resolve()` + parent check | `tools.py` + `documents.py` | Directory traversal |
| `SecretStr` | `config.py` | API key leakage in logs/repr |
| Tools return strings | `tools.py` | Agent loop crashes on tool errors |
| Typed exception hierarchy | `chat.py` | Stack traces in HTTP responses |
| Double-layered checks | Both API and tool layer | Defense in depth |

---

Next: [Chapter 10 → Running & Extending](./10_running_and_extending.md)
