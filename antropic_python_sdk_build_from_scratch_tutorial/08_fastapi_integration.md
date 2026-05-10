# Chapter 08 — FastAPI Integration

> **Goal:** Wire everything together into a working HTTP API — the application factory, lifespan hooks, routers, Pydantic schemas, and endpoint handlers.

---

## 8.1 Pydantic Schemas (Presentation Contracts)

Schemas define the shape of HTTP request/response bodies.  They live in `app/schemas/` and are **only** used by the presentation layer — they must not be imported by domain or service code.

### Chat Schemas

```python
# app/schemas/chat.py
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique conversation identifier. Reuse across turns to maintain history.",
        examples=["user-123-session-abc"],
    )
    user_message: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="The user's natural language prompt or question.",
        examples=["Please summarize sample_report.txt"],
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Claude's final answer.")
    status: str = Field(
        default="success",
        description="'success' on normal completion, 'error' if the agent loop failed.",
    )
```

### Document Schemas

```python
# app/schemas/documents.py
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file.")
    size_bytes: int = Field(..., description="Size of the uploaded file in bytes.")
    status: str = Field(default="uploaded")


class DeleteResponse(BaseModel):
    filename: str = Field(..., description="Name of the deleted file.")
    status: str = Field(default="deleted")


class ListResponse(BaseModel):
    documents: list[str] = Field(..., description="Sorted list of available document filenames.")
    total: int = Field(..., description="Total number of documents.")
```

---

## 8.2 The Chat Endpoint

The chat handler is intentionally **thin**: deserialise → delegate → serialise.  No business logic here.

```python
# app/api/v1/chat.py
import logging

from fastapi import APIRouter

from app.domain.exceptions import AgentError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent import agent_service
from app.services.memory import session_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the AI Agent",
    description=(
        "Submit a natural language message. The agent will autonomously decide "
        "which local-file tools to call, loop until it has enough context, and "
        "return a final summarised answer."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info(
        "Chat request | session='%s' | message='%.80s...'",
        request.session_id,
        request.user_message,
    )

    try:
        session = session_store.get_or_create(request.session_id)
        session.add_user_message(request.user_message)

        answer = await agent_service.run(session)

        session_store.save(session)
        logger.info("Chat success | session='%s'", request.session_id)
        return ChatResponse(response=answer, status="success")

    except AgentError as exc:
        # Known domain error — log as warning, return user-safe message
        logger.warning("Agent error for session '%s': %s", request.session_id, exc)
        return ChatResponse(
            response="The agent encountered an error and could not complete your request.",
            status="error",
        )
    except Exception as exc:
        # Unexpected error — log with full traceback, never expose internals
        logger.error(
            "Unhandled error for session '%s': %s",
            request.session_id,
            exc,
            exc_info=True,
        )
        return ChatResponse(
            response="An unexpected internal error occurred. Please try again.",
            status="error",
        )
```

**Why two `except` clauses?**

| Exception type | Meaning | Response |
|---|---|---|
| `AgentError` | Known, typed domain error | User-safe message, `status="error"` |
| `Exception` | Unknown / unexpected error | Generic message + full traceback in logs |

Never return stack traces, SQL errors, or file paths to the client.

---

## 8.3 The Documents Endpoint

Document management (upload, list, delete) is a separate router with its own security checks:

```python
# app/api/v1/documents.py (key sections)
import logging
import re
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.schemas.documents import DeleteResponse, ListResponse, UploadResponse
from app.services.tools import DOCS_DIR

router = APIRouter()
logger = logging.getLogger(__name__)

_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-. ]*$")
_ALLOWED_EXTENSIONS = {".txt", ".md", ".csv"}


def _validate_filename(filename: str) -> str:
    """Sanitise filename; raise HTTPException on failure."""
    # Strip any path components (e.g., "../../etc/passwd" → "passwd" → still rejected by regex)
    safe_name = PurePosixPath(filename).name

    if not safe_name:
        raise HTTPException(status_code=400, detail="Filename must not be empty.")

    if not _SAFE_FILENAME_RE.match(safe_name):
        raise HTTPException(
            status_code=400,
            detail=f"Filename '{safe_name}' contains invalid characters.",
        )

    ext = Path(safe_name).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Extension '{ext}' is not supported. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    return safe_name


@router.get("", response_model=ListResponse, summary="List all available documents")
async def list_documents() -> ListResponse:
    files = sorted(f.name for f in DOCS_DIR.iterdir() if f.is_file())
    return ListResponse(documents=files, total=len(files))


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new document",
)
async def upload_document(file: UploadFile) -> UploadResponse:
    safe_name = _validate_filename(file.filename or "")
    target = (DOCS_DIR / safe_name).resolve()

    if target.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Document '{safe_name}' already exists. Delete it first to replace.",
        )

    content = await file.read()
    try:
        content.decode("utf-8")  # validate UTF-8 before writing
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File must be valid UTF-8 encoded text.")

    target.write_bytes(content)
    logger.info("Document uploaded: '%s' (%d bytes)", safe_name, len(content))
    return UploadResponse(filename=safe_name, size_bytes=len(content))


@router.delete("/{filename}", response_model=DeleteResponse, summary="Delete a document")
async def delete_document(filename: str) -> DeleteResponse:
    safe_name = _validate_filename(filename)
    target = (DOCS_DIR / safe_name).resolve()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Document '{safe_name}' not found.")

    target.unlink()
    logger.info("Document deleted: '%s'", safe_name)
    return DeleteResponse(filename=safe_name)
```

---

## 8.4 The Application Factory

```python
# app/main.py
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router
from app.core.config import settings


def _configure_logging() -> None:
    log_level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan — startup / shutdown hooks."""
    _configure_logging()
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting AI Agent MVP | env=%s | model=%s",
        settings.app_env,
        settings.claude_model,
    )
    yield
    logger.info("Shutting down AI Agent MVP.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Local File Reader & Summarizer AI Agent",
        description="MVP AI Agent that reads local documents and answers questions using Claude.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — open for local development; tighten for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])

    @app.get("/health", tags=["ops"], summary="Liveness probe")
    async def health():
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()
```

### Key decisions in `main.py`

**App factory pattern (`create_app()`):** instead of a module-level `app = FastAPI()`, wrapping in a function makes it easy to create test instances with different settings.

**`lifespan` context manager (not `@app.on_event`):** FastAPI deprecated `on_event` in favour of the `lifespan` parameter. The `@asynccontextmanager` approach is cleaner and supports both startup and shutdown in one function.

**Logging in lifespan, not at module level:** configuration runs after the FastAPI app is created, ensuring log output appears with the correct format from the first request.

---

## 8.5 The Complete URL Map

| Method | Path | Handler | Description |
|--------|------|---------|-------------|
| `GET` | `/health` | `health()` | Liveness probe |
| `POST` | `/api/v1/chat` | `chat()` | Send message to agent |
| `GET` | `/api/v1/documents` | `list_documents()` | List all documents |
| `POST` | `/api/v1/documents` | `upload_document()` | Upload a new document |
| `DELETE` | `/api/v1/documents/{filename}` | `delete_document()` | Delete a document |
| `GET` | `/docs` | FastAPI built-in | Swagger UI |
| `GET` | `/redoc` | FastAPI built-in | ReDoc UI |

---

## Chapter Summary

| Concept | Key Takeaway |
|---|---|
| Thin handlers | Deserialise → delegate to service → serialise. No business logic. |
| Two `except` clauses | Typed domain errors vs unexpected errors — never expose internals |
| App factory | `create_app()` function for testability |
| Lifespan context manager | Preferred over deprecated `@app.on_event` |
| Pydantic schemas | Presentation layer only — never imported by domain/services |

---

Next: [Chapter 09 → Security Patterns](./09_security.md)
