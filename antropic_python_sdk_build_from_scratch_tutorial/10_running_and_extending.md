# Chapter 10 — Running & Extending

> **Goal:** Start the server, make real API calls, understand the logs, and know exactly how to extend the agent with new capabilities.

---

## 10.1 Start the Backend

```bash
cd ai-agent-mvp

# Option A: uv (recommended)
uv run uvicorn app.main:app --reload --port 8000

# Option B: plain Python (activate venv first)
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**Expected startup output:**
```
2024-03-10T09:00:00 | INFO     | app.main | Starting AI Agent MVP | env=development | model=claude-3-7-sonnet-20250219
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
```

**Interactive API docs:**
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 10.2 Essential curl Commands

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok","env":"development"}
```

### List Documents

```bash
curl http://localhost:8000/api/v1/documents
# {"documents":["getting_started.md","sample_report.txt"],"total":2}
```

### Upload a Document

```bash
# Create a test file
echo "Q3 Revenue: \$4.2M. Region: West Coast exceeded target by 12%." > /tmp/q3_report.txt

# Upload it
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@/tmp/q3_report.txt"
# {"filename":"q3_report.txt","size_bytes":62,"status":"uploaded"}
```

### Chat — Single Turn

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "user_message": "What documents do you have?"}'
# {"response":"I have access to the following documents:\n- getting_started.md\n- q3_report.txt","status":"success"}
```

### Chat — Ask About a Document

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "user_message": "Summarize q3_report.txt"}'
```

### Chat — Multi-Turn Follow-Up

```bash
# Same session_id preserves history from previous turns
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "user_message": "Which region exceeded its target?"}'
# Claude remembers the report from the previous turn — no re-reading needed
```

### Delete a Document

```bash
curl -X DELETE http://localhost:8000/api/v1/documents/q3_report.txt
# {"filename":"q3_report.txt","status":"deleted"}
```

---

## 10.3 Reading the Logs

When `DEBUG=true` in `.env`, you can trace every agent loop iteration:

```
2024-03-10T09:01:22 | INFO     | app.api.v1.chat | Chat request | session='demo-1' | message='Summarize q3_report.txt...'
2024-03-10T09:01:22 | DEBUG    | app.services.agent | Agent loop iteration 0 for session 'demo-1'
2024-03-10T09:01:23 | DEBUG    | app.services.agent | Claude stop_reason=tool_use
2024-03-10T09:01:23 | INFO     | app.services.agent | Executing tool 'list_local_documents' with input: {}
2024-03-10T09:01:23 | DEBUG    | app.services.agent | Agent loop iteration 1 for session 'demo-1'
2024-03-10T09:01:24 | DEBUG    | app.services.agent | Claude stop_reason=tool_use
2024-03-10T09:01:24 | INFO     | app.services.agent | Executing tool 'read_local_document' with input: {'file_name': 'q3_report.txt'}
2024-03-10T09:01:24 | INFO     | app.services.tools | Read file 'q3_report.txt' (62 chars)
2024-03-10T09:01:24 | DEBUG    | app.services.agent | Agent loop iteration 2 for session 'demo-1'
2024-03-10T09:01:26 | DEBUG    | app.services.agent | Claude stop_reason=end_turn
2024-03-10T09:01:26 | INFO     | app.api.v1.chat | Chat success | session='demo-1'
```

**Log format explanation:**

| Field | Example | Meaning |
|-------|---------|---------|
| Timestamp | `2024-03-10T09:01:22` | When the event occurred |
| Level | `INFO`, `DEBUG` | Severity |
| Logger | `app.services.agent` | Which module logged this |
| Message | `Agent loop iteration 0` | What happened |

---

## 10.4 Adding a New Tool (Complete Walkthrough)

Let's add a `word_count_document` tool that counts words in a file.

### Step 1: Implement the function in `app/services/tools.py`

```python
def word_count_document(file_name: str) -> str:
    """Return the word count and line count of file_name."""
    content = read_local_document(file_name)
    if content.startswith("Error:"):
        return content  # propagate error strings

    lines = content.splitlines()
    words = content.split()
    return (
        f"File '{file_name}' contains {len(words)} words across {len(lines)} lines."
    )
```

### Step 2: Add its schema to `TOOL_DEFINITIONS`

```python
TOOL_DEFINITIONS.append({
    "name": "word_count_document",
    "description": (
        "Returns the word count and line count of a specific document. "
        "Use this when the user asks how long a document is, or before "
        "deciding whether to read the full content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string",
                "description": "The exact name of the file to count, including extension.",
            }
        },
        "required": ["file_name"],
    },
})
```

### Step 3: Register it in `TOOL_REGISTRY`

```python
TOOL_REGISTRY["word_count_document"] = lambda inp: word_count_document(inp["file_name"])
```

### Test it

```bash
uv run uvicorn app.main:app --reload --port 8000

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-wc", "user_message": "How many words are in q3_report.txt?"}'
# {"response":"The file 'q3_report.txt' contains 12 words across 1 lines.","status":"success"}
```

**`AgentService` was never modified.** ✓

---

## 10.5 Running Tests

```bash
# Run all tests
uv run pytest -v

# Run only domain tests (no API key needed)
uv run pytest tests/test_domain.py -v

# Run with coverage
uv run pytest --cov=app --cov-report=term-missing
```

**Sample test for the new word_count tool:**

```python
# tests/test_tools.py
from app.services.tools import word_count_document, DOCS_DIR


def test_word_count_document(tmp_path, monkeypatch):
    # Point DOCS_DIR to a temp directory for testing
    monkeypatch.setattr("app.services.tools.DOCS_DIR", tmp_path)
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello world this is a test\nSecond line here")

    result = word_count_document("test.txt")
    assert "9 words" in result
    assert "2 lines" in result


def test_word_count_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.tools.DOCS_DIR", tmp_path)
    result = word_count_document("nonexistent.txt")
    assert result.startswith("Error:")
```

---

## 10.6 Running in Production

For production deployment, make these changes:

### Tighten CORS

```python
# app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # not "*"
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
```

### Use Multiple Workers

```bash
uvicorn app.main:app --workers 4 --port 8000 --host 0.0.0.0
```

> ⚠️ With multiple workers, the `InMemorySessionStore` is **not shared** between worker processes. Upgrade to Redis before adding workers.

### Set `DEBUG=false`

```bash
# .env
DEBUG=false
APP_ENV=production
```

### Use Gunicorn + Uvicorn Workers (Best Practice)

```bash
pip install gunicorn
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 10.7 Common Issues & Solutions

| Problem | Likely Cause | Solution |
|---------|-------------|---------|
| `ANTHROPIC_API_KEY` validation error on startup | `.env` not found or key missing | Run `cp .env.example .env` and fill in key |
| `400 Bad Request` on file upload | Extension not in `{".txt", ".md", ".csv"}` | Only upload supported formats |
| `409 Conflict` on file upload | File already exists | Delete it first with `DELETE /api/v1/documents/{name}` |
| Agent returns "The agent encountered an error" | `AgentLoopError` (loop cap hit) | Check logs; reduce complexity or increase `_MAX_LOOP_ITERATIONS` |
| Session history not preserved | Different `session_id` between requests | Reuse the exact same `session_id` string across turns |
| Slow first response | Cold SDK connection | Subsequent requests are faster (connection pooling) |

---

## 10.8 Future Extension Ideas

The MVP is designed to grow.  Here are natural next steps:

| Feature | What to change |
|---------|---------------|
| **Streaming responses** | Replace `messages.create()` with `messages.stream()` + SSE endpoint |
| **PDF support** | Add a `read_pdf_document` tool using `pypdf` |
| **Persistent sessions** | Swap `InMemorySessionStore` → `RedisSessionStore` (only `memory.py`) |
| **Vector search (RAG)** | Add `search_semantic` tool backed by `chromadb` or `pgvector` |
| **User authentication** | Add FastAPI `Depends` auth middleware in `api/v1/` |
| **Rate limiting** | Add `slowapi` middleware to `main.py` |
| **New transport** | Add a CLI interface that calls `agent_service.run()` directly |

---

## Final Architecture at a Glance

```
Client
  │
  ├─ POST /api/v1/chat ──▶ chat.py ──▶ session_store ──▶ agent_service.run()
  │                                                              │
  │                                                    ┌─────────▼────────┐
  │                                                    │   AGENTIC LOOP   │
  │                                                    │  AsyncAnthropic  │
  │                                                    │  .messages.create│
  │                                                    └─────────┬────────┘
  │                                                    stop_reason == tool_use
  │                                                              │
  │                                                    execute_tool(name, input)
  │                                                    ← "Error: ..." or content
  │                                                    feed back as tool_result
  │                                                    stop_reason == end_turn
  │                                                              │
  └─ ChatResponse(response, status) ◀────────────────── return final_text
  
  ├─ GET/POST/DELETE /api/v1/documents ──▶ documents.py ──▶ pathlib (sandboxed)
  │
  └─ GET /health ──▶ {"status": "ok"}
```

---

## Congratulations!

You have built a production-quality AI Agent from scratch:

- ✅ **Anthropic SDK** — `AsyncAnthropic`, `messages.create()`, tool schemas
- ✅ **Agentic loop** — `stop_reason` → `tool_use` → `tool_result` cycle
- ✅ **Clean Architecture** — Domain / Application / Infrastructure / Presentation
- ✅ **Tool system** — JSON Schema definitions, registry, Open/Closed pattern
- ✅ **Session management** — multi-turn conversation with swappable backend
- ✅ **Security** — path traversal prevention, `SecretStr`, graceful degradation
- ✅ **FastAPI** — app factory, lifespan, routers, Pydantic schemas

---

← [Back to README](./README.md)
