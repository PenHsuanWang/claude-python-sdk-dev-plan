# Chapter 03 — Clean Architecture Overview

> **Goal:** Understand why the project is structured the way it is, and how the four layers relate to each other. This mental model makes every subsequent chapter easier to follow.

---

## 3.1 The Four Layers

```
┌──────────────────────────────────────────────────┐
│  PRESENTATION LAYER  (app/api/, app/schemas/)     │
│  FastAPI routers, Pydantic request/response models│
│  HTTP-specific concerns only                      │
├──────────────────────────────────────────────────┤
│  APPLICATION LAYER   (app/services/agent.py)      │
│  Business logic — the agentic loop               │
│  Orchestrates Domain + Infrastructure             │
├──────────────────────────────────────────────────┤
│  INFRASTRUCTURE LAYER (app/services/tools.py,    │
│                        app/services/memory.py)    │
│  File system access, session store, SDK calls    │
├──────────────────────────────────────────────────┤
│  DOMAIN LAYER        (app/domain/)                │
│  Pure Python dataclasses + exceptions             │
│  ZERO external dependencies                       │
└──────────────────────────────────────────────────┘
```

**The Dependency Rule:** dependencies only point **inward**.

```
Presentation → Application → Domain
Infrastructure → Domain
Application → Infrastructure (via abstractions)
```

> **What this means in practice:** The `app/domain/` folder has NO imports from FastAPI, Anthropic, or pydantic. You can test it with zero mocking.

---

## 3.2 Layer-by-Layer File Map

### Domain Layer (innermost — no external imports)

```
app/domain/
├── models.py       # AgentSession dataclass
└── exceptions.py   # AgentError, AgentLoopError, ToolSecurityError, ...
```

These files only use Python's standard library (`dataclasses`, `typing`). They express **what the system is** in pure business concepts.

### Application Layer

```
app/services/
└── agent.py        # AgentService — the agentic loop
```

`AgentService` knows about `AgentSession` (domain) and calls `execute_tool()` (infrastructure) and `_client.messages.create()` (SDK). It has **no knowledge of HTTP, sessions IDs, or file paths**.

### Infrastructure Layer

```
app/services/
├── tools.py        # list_local_documents, read_local_document, TOOL_DEFINITIONS
└── memory.py       # InMemorySessionStore
```

These interact with the real world (filesystem, in-memory dict) but expose clean interfaces to the application layer.

### Presentation Layer (outermost)

```
app/
├── main.py         # FastAPI app factory + lifespan
├── schemas/
│   ├── chat.py     # ChatRequest, ChatResponse
│   └── documents.py # ListResponse, UploadResponse, DeleteResponse
└── api/v1/
    ├── chat.py     # POST /api/v1/chat
    └── documents.py # GET|POST|DELETE /api/v1/documents
```

These files know about HTTP status codes, multipart uploads, and JSON. They must NOT contain business logic — they delegate to `AgentService`.

---

## 3.3 Data Flow for a Chat Request

```
POST /api/v1/chat
{"session_id": "u1", "user_message": "Summarize report.txt"}
        │
        ▼
[PRESENTATION]  chat.py handler
  - Deserialise ChatRequest (Pydantic)
  - Load session from session_store.get_or_create("u1")
  - session.add_user_message("Summarize report.txt")
  - answer = await agent_service.run(session)
        │
        ▼
[APPLICATION]   AgentService.run(session)
  - Loop: call Claude with session.messages + TOOL_DEFINITIONS
  - response.stop_reason == "tool_use" → call execute_tool(...)
  - response.stop_reason == "end_turn" → return final text
        │
        ▼
[INFRASTRUCTURE] execute_tool("read_local_document", {"file_name": "report.txt"})
  - Resolve path safely (pathlib)
  - Read file content
  - Return string to AgentService
        │
        ▼ (back up the chain)
[PRESENTATION]  Return ChatResponse(response=answer, status="success")
```

---

## 3.4 SOLID Principles in Practice

### Single Responsibility

- `chat.py` handler: only HTTP concerns (deserialise → delegate → serialise)
- `AgentService`: only loop orchestration
- `tools.py`: only file I/O

### Open/Closed Principle

To add a new tool you only touch `tools.py`:
1. Write a new function
2. Add its schema to `TOOL_DEFINITIONS`
3. Register it in `TOOL_REGISTRY`

**`AgentService` is never modified.** It dispatches dynamically via `execute_tool()`.

### Dependency Inversion

`AgentService` depends on the abstract function `execute_tool(name, input) → str`, not on any specific tool implementation.  Similarly, swapping `InMemorySessionStore` for a Redis-backed store requires only changing `memory.py`.

---

## 3.5 Why Not Put Everything in One File?

A common shortcut is to put all logic in one big `main.py`.  Here is why that becomes painful:

| Monolithic | Layered |
|---|---|
| Testing the agent loop requires a real FastAPI server | `AgentService` can be unit-tested with a mock client |
| Adding a new transport (WebSocket, CLI) means duplicating logic | Just add a new Presentation entry point |
| A bug in file I/O is hard to isolate | `tools.py` is independently testable |
| Changing the session backend touches multiple files | Only `memory.py` needs to change |

---

## Chapter Summary

| Concept | Key Takeaway |
|---|---|
| 4 layers | Domain → Infrastructure/Application → Presentation |
| Dependency rule | Inner layers never import outer layers |
| Domain layer | Pure Python, zero deps, perfectly unit-testable |
| Open/Closed | New tools only touch `tools.py` — agent loop never changes |

---

Next: [Chapter 04 → Domain Layer](./04_domain_layer.md)
