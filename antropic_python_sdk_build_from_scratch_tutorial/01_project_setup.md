# Chapter 01 — Project Setup

> **Goal:** Create a clean Python project with all the dependencies, environment configuration, and folder structure you need before writing a single line of agent code.

---

## 1.1 Why `uv`?

This project uses [`uv`](https://docs.astral.sh/uv/) — a blazing-fast Python package manager written in Rust.  It combines `pip`, `venv`, and `pip-tools` into one tool and resolves dependencies in milliseconds.

```bash
# Install uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

> **Alternatively** you can use plain `pip` + `venv`.  Every `uv run` command below has a `python` equivalent.

---

## 1.2 Initialise the Project

```bash
mkdir ai-agent-mvp && cd ai-agent-mvp

# Create the virtual environment and pyproject.toml in one step
uv init --python 3.12
```

Replace the generated `pyproject.toml` with this production-ready version:

```toml
# pyproject.toml
[project]
name = "ai-agent-mvp"
version = "0.1.0"
description = "Local File Reader & Summarizer AI Agent (MVP)"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",          # web framework
    "uvicorn[standard]>=0.30.0", # ASGI server
    "anthropic>=0.40.0",         # Anthropic Python SDK ← the star of this tutorial
    "pydantic>=2.0.0",           # data validation
    "pydantic-settings>=2.0.0",  # .env → typed config
    "python-dotenv>=1.0.0",      # load .env automatically
    "python-multipart>=0.0.9",   # file upload support
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",     # async HTTP test client for FastAPI
    "ruff>=0.4.0",       # linter + formatter
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

Install everything:

```bash
uv sync          # creates .venv and installs all dependencies
uv sync --dev    # also installs dev dependencies
```

---

## 1.3 Create the Folder Structure

```bash
mkdir -p app/core app/domain app/schemas app/api/v1 app/services
mkdir -p local_data

# Touch all __init__.py files
touch app/__init__.py
touch app/core/__init__.py
touch app/domain/__init__.py
touch app/schemas/__init__.py
touch app/api/__init__.py
touch app/api/v1/__init__.py
touch app/services/__init__.py
```

---

## 1.4 Environment Variables

Create `.env.example` — this is the file you commit to version control:

```bash
# .env.example

# ─── Anthropic ───────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: override for corporate LLM gateway / Zero Data Retention proxy
# ANTHROPIC_BASE_URL=https://your-internal-gateway.example.com

# ─── Model ───────────────────────────────────────────────────────────────────
CLAUDE_MODEL=claude-3-7-sonnet-20250219
MAX_TOKENS=4096
MAX_RETRIES=2

# ─── App ─────────────────────────────────────────────────────────────────────
APP_ENV=development
DEBUG=true
LOCAL_DATA_DIR=local_data
```

Copy it and fill in your real key:

```bash
cp .env.example .env
# edit .env → set ANTHROPIC_API_KEY=sk-ant-YOUR-REAL-KEY
```

Add `.env` to `.gitignore` **immediately**:

```bash
echo ".env" >> .gitignore
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "local_data/*.txt" >> .gitignore   # optional: don't commit sample data
```

---

## 1.5 The Configuration Module

`pydantic-settings` reads environment variables (or `.env`) and validates them into a typed Python object.  Using `SecretStr` for the API key ensures it **never appears in logs, repr() output, or JSON serialization**.

```python
# app/core/config.py
"""Core configuration — strong-typed, validated settings from environment."""

from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Anthropic credentials ─────────────────────────────────────────────── #
    anthropic_api_key: SecretStr = Field(..., alias="ANTHROPIC_API_KEY")
    anthropic_base_url: Optional[str] = Field(None, alias="ANTHROPIC_BASE_URL")

    # ── Model parameters ──────────────────────────────────────────────────── #
    claude_model: str = Field("claude-3-7-sonnet-20250219", alias="CLAUDE_MODEL")
    max_tokens: int = Field(4096, alias="MAX_TOKENS")
    max_retries: int = Field(2, alias="MAX_RETRIES")

    # ── Application ───────────────────────────────────────────────────────── #
    app_env: str = Field("development", alias="APP_ENV")
    debug: bool = Field(False, alias="DEBUG")
    local_data_dir: str = Field("local_data", alias="LOCAL_DATA_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",   # silently ignore unknown env vars
    )


# Module-level singleton — import `settings` everywhere instead of re-parsing
settings = Settings()
```

**Key design choices:**
- `Field(..., alias="ANTHROPIC_API_KEY")` — the Python attribute is snake_case but the env var is SCREAMING_SNAKE_CASE.
- `SecretStr` — calling `settings.anthropic_api_key` gives you a `SecretStr` object, NOT the raw string. Call `.get_secret_value()` only when you actually need to pass it to the SDK.
- The singleton `settings = Settings()` is created once at import time. Import `settings` everywhere — never instantiate `Settings()` again.

**Verify it works:**

```bash
uv run python -c "from app.core.config import settings; print(settings.claude_model)"
# → claude-3-7-sonnet-20250219
```

---

## 1.6 Add a Sample Document

```bash
cat > local_data/getting_started.md << 'EOF'
# Getting Started

Welcome to the AI Agent MVP.  This document lives in the sandboxed `local_data/`
directory.  The agent can list, read, and summarize files in this folder.

## Supported Formats
- Plain text (.txt)
- Markdown (.md)
- CSV (.csv)

## How to use
Send a POST request to /api/v1/chat with a natural language question.
The agent will autonomously decide which files to read and return a summary.
EOF
```

---

## Chapter Summary

| What you did | Why |
|---|---|
| Used `uv` to manage the project | Fast, reproducible dependency management |
| Defined all deps in `pyproject.toml` | Single source of truth for the project |
| Created the folder structure upfront | Enforces clean architecture from day one |
| Used `SecretStr` for the API key | Prevents accidental secret exposure |
| Created a module-level `settings` singleton | Fast startup; no repeated env file parsing |

---

Next: [Chapter 02 → Anthropic SDK Basics](./02_anthropic_sdk_basics.md)
