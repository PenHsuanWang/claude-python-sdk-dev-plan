# Data Scientist AI-Agent — From Scratch

> **A domain-aware agentic system that reasons about data with physical meaning**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Anthropic SDK](https://img.shields.io/badge/anthropic--sdk-0.50%2B-blueviolet.svg)](https://github.com/anthropic/anthropic-python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project teaches you how to build a **production-quality AI agent for data science** from absolute first principles — no LangChain, no AutoGen, no magic. Starting from a minimal FastAPI + Anthropic SDK skeleton, you extend it step-by-step into a domain-aware data scientist agent capable of loading datasets, executing Python code, validating physical units, and exporting reproducible Jupyter notebooks.

The central thesis is that **data science results must be physically meaningful**, not just numerically plausible. An agent that computes "thermal efficiency = 112.5%" has made a catastrophic error — it violates the first law of thermodynamics — yet most AI systems have no mechanism to catch it. This agent does, by pairing code execution with a physical validation layer backed by domain documents and the `pint` unit library.

The guide is structured as a progressive learning path: you begin with fundamentals (ReAct, Clean Architecture, physical modeling), move through the Anthropic SDK capabilities that power the system, implement each component incrementally, and finish with a reference section and worked examples you can run against real datasets.

---

## Key Differentiators

- **Explicit ReAct loop** — Every reasoning step is visible as `Thought → Action → Observation` text, giving full auditability instead of black-box tool-chaining
- **Physical unit validation** — The `pint` library enforces dimensional correctness; domain-specific range checks flag results that violate physics (efficiency > 100%, negative absolute temperatures, etc.)
- **Domain knowledge injection** — Before any analysis, the agent reads relevant domain documents and extracts unit definitions, typical operating ranges, and physical laws relevant to the dataset
- **Three code execution backends** — Local subprocess, Jupyter kernel (persistent state), or Anthropic's cloud sandbox (`code_execution_20260120`) — switchable via config
- **Reproducible notebook export** — Every code cell and output accumulated during a session is exported as a `.ipynb` file you can run, share, or version-control
- **Native Anthropic SDK** — Zero third-party orchestration; you see exactly how tool definitions, `tool_use` blocks, and `tool_result` blocks work at the API level

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                           │
│   FastAPI  ─  POST /api/v1/analysis/chat                            │
│              GET  /api/v1/analysis/{id}/notebook                    │
│              GET  /api/v1/datasets                                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ depends on
┌───────────────────────────▼─────────────────────────────────────────┐
│                       APPLICATION LAYER                             │
│   DataScienceAgentService  (ReAct loop orchestration)               │
│   ├── ToolExecutor          (dispatches 14 tools)                   │
│   ├── PhysicalValidator     (pint + domain ranges)                  │
│   └── NotebookExporter      (nbformat cell accumulation)            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ depends on
┌───────────────────────────▼─────────────────────────────────────────┐
│                     INFRASTRUCTURE LAYER                            │
│   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│   │  Code Execution  │  │  Document Store   │  │  Dataset Store   │  │
│   │  ─────────────  │  │  ─────────────── │  │  ─────────────── │  │
│   │  Subprocess      │  │  LocalFileStore  │  │  PandasLoader    │  │
│   │  JupyterKernel   │  │  (markdown/pdf)  │  │  (csv/xls/h5/   │  │
│   │  AnthropicCloud  │  │                  │  │   parquet/json)  │  │
│   └─────────────────┘  └──────────────────┘  └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ depends on
┌───────────────────────────▼─────────────────────────────────────────┐
│                         DOMAIN LAYER                                │
│   AnalysisSession · DatasetMeta · AnalysisResult · PhysicalUnit     │
│   DomainProfile · ExecutionResult · NotebookCell                    │
│   (Pure Python dataclasses — zero framework dependencies)           │
└─────────────────────────────────────────────────────────────────────┘
                            │ calls
┌───────────────────────────▼─────────────────────────────────────────┐
│                      ANTHROPIC CLAUDE API                           │
│   claude-opus-4-7 / claude-sonnet-4-6 / claude-haiku-4-5           │
│   Tools: user-defined (14) + server (code_execution_20260120)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

### Fundamentals
1. [01_fundamentals/01_react_pattern.md](01_fundamentals/01_react_pattern.md) — ReAct: Reason + Act loop, parsing, pitfalls
2. [01_fundamentals/02_clean_architecture.md](01_fundamentals/02_clean_architecture.md) — Four-layer architecture, SOLID, DI
3. [01_fundamentals/03_physical_domain_modeling.md](01_fundamentals/03_physical_domain_modeling.md) — pint, dimensional analysis, domain profiles
4. [01_fundamentals/04_data_science_workflow.md](01_fundamentals/04_data_science_workflow.md) — CRISP-DM + AI augmentation, EDA checklist

### Anthropic SDK Reference
5. [02_anthropic_sdk_reference/01_client_setup.md](02_anthropic_sdk_reference/01_client_setup.md) — Client initialization, auth, retries
6. [02_anthropic_sdk_reference/02_messages_api.md](02_anthropic_sdk_reference/02_messages_api.md) — Messages API deep-dive, stop_reasons
7. [02_anthropic_sdk_reference/03_tool_definitions.md](02_anthropic_sdk_reference/03_tool_definitions.md) — Tool schemas, strict mode, input validation
8. [02_anthropic_sdk_reference/04_tool_use_loop.md](02_anthropic_sdk_reference/04_tool_use_loop.md) — tool_use → tool_result protocol
9. [02_anthropic_sdk_reference/05_server_tools.md](02_anthropic_sdk_reference/05_server_tools.md) — code_execution, web_search, Files API
10. [02_anthropic_sdk_reference/06_model_selection.md](02_anthropic_sdk_reference/06_model_selection.md) — Model comparison, cost, context windows
11. [02_anthropic_sdk_reference/07_streaming.md](02_anthropic_sdk_reference/07_streaming.md) — Streaming responses, token events
12. [02_anthropic_sdk_reference/08_error_handling.md](02_anthropic_sdk_reference/08_error_handling.md) — API errors, retries, timeouts

### Software Design
13. [03_software_design/01_domain_models.md](03_software_design/01_domain_models.md) — All dataclasses: AnalysisSession, DatasetMeta, etc.
14. [03_software_design/02_tool_catalog.md](03_software_design/02_tool_catalog.md) — All 14 tools: signature, schema, return format
15. [03_software_design/03_tool_execution_engine.md](03_software_design/03_tool_execution_engine.md) — ToolExecutor dispatch, error handling
16. [03_software_design/04_code_execution_backends.md](03_software_design/04_code_execution_backends.md) — Subprocess, Jupyter, Anthropic backends
17. [03_software_design/05_physical_validation_engine.md](03_software_design/05_physical_validation_engine.md) — pint integration, range checks, law checks
18. [03_software_design/06_notebook_exporter.md](03_software_design/06_notebook_exporter.md) — nbformat, cell accumulation, export
19. [03_software_design/07_agent_service.md](03_software_design/07_agent_service.md) — DataScienceAgentService, ReAct orchestration
20. [03_software_design/08_api_design.md](03_software_design/08_api_design.md) — New endpoints, request/response schemas

### Implementation Guide
21. [04_implementation_guide/01_project_setup.md](04_implementation_guide/01_project_setup.md) — uv, pyproject.toml, env vars
22. [04_implementation_guide/02_domain_layer.md](04_implementation_guide/02_domain_layer.md) — Implement all domain models
23. [04_implementation_guide/03_infrastructure_layer.md](04_implementation_guide/03_infrastructure_layer.md) — File store, dataset loader
24. [04_implementation_guide/04_code_execution.md](04_implementation_guide/04_code_execution.md) — All three backends end-to-end
25. [04_implementation_guide/05_tool_implementations.md](04_implementation_guide/05_tool_implementations.md) — All 14 tools, tested
26. [04_implementation_guide/06_agent_service_impl.md](04_implementation_guide/06_agent_service_impl.md) — Full ReAct loop implementation
27. [04_implementation_guide/07_api_layer.md](04_implementation_guide/07_api_layer.md) — FastAPI routes, middleware
28. [04_implementation_guide/08_testing.md](04_implementation_guide/08_testing.md) — pytest, fixtures, integration tests

### Reference
29. [05_reference/01_configuration_reference.md](05_reference/01_configuration_reference.md) — All Settings fields
30. [05_reference/02_tool_api_reference.md](05_reference/02_tool_api_reference.md) — Tool quick-reference card
31. [05_reference/03_troubleshooting.md](05_reference/03_troubleshooting.md) — Common errors and fixes
32. [05_reference/04_worked_example.md](05_reference/04_worked_example.md) — Full walkthrough with power plant dataset

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/claude_python_sdk.git
cd ai-agent-from-scratch/data-scientist-agent
uv sync

# 2. Set API key and start server
export ANTHROPIC_API_KEY=sk-ant-...
uv run uvicorn app.main:app --reload --port 8001

# 3. Send an analysis request
curl -X POST http://localhost:8001/api/v1/analysis/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze the turbine_data.csv dataset and validate all efficiency values", "session_id": null}'
```

---

## Document Structure

| File | Layer | Purpose |
|------|-------|---------|
| `README.md` | — | Project overview and navigation |
| `00_motivation_and_goals.md` | — | Why this system, design goals, alternatives |
| `01_fundamentals/01_react_pattern.md` | Concept | ReAct loop theory and implementation |
| `01_fundamentals/02_clean_architecture.md` | Concept | Architecture layers and SOLID |
| `01_fundamentals/03_physical_domain_modeling.md` | Concept | pint, units, domain validation |
| `01_fundamentals/04_data_science_workflow.md` | Concept | CRISP-DM + AI augmentation |
| `02_anthropic_sdk_reference/01_client_setup.md` | SDK | Client, auth, retries |
| `02_anthropic_sdk_reference/02_messages_api.md` | SDK | Messages API parameters |
| `02_anthropic_sdk_reference/03_tool_definitions.md` | SDK | Tool schema format |
| `02_anthropic_sdk_reference/04_tool_use_loop.md` | SDK | Tool call/result protocol |
| `02_anthropic_sdk_reference/05_server_tools.md` | SDK | code_execution, Files API |
| `02_anthropic_sdk_reference/06_model_selection.md` | SDK | Models, costs, context |
| `02_anthropic_sdk_reference/07_streaming.md` | SDK | Streaming events |
| `02_anthropic_sdk_reference/08_error_handling.md` | SDK | Errors and retries |
| `03_software_design/01_domain_models.md` | Domain | Dataclass definitions |
| `03_software_design/02_tool_catalog.md` | Application | All 14 tool specs |
| `03_software_design/03_tool_execution_engine.md` | Application | Dispatcher design |
| `03_software_design/04_code_execution_backends.md` | Infrastructure | Three runners |
| `03_software_design/05_physical_validation_engine.md` | Application | Validation engine |
| `03_software_design/06_notebook_exporter.md` | Infrastructure | Notebook generation |
| `03_software_design/07_agent_service.md` | Application | ReAct orchestrator |
| `03_software_design/08_api_design.md` | Presentation | API routes |
| `04_implementation_guide/01_project_setup.md` | — | Setup steps |
| `04_implementation_guide/02_domain_layer.md` | Domain | Implement models |
| `04_implementation_guide/03_infrastructure_layer.md` | Infrastructure | File/data stores |
| `04_implementation_guide/04_code_execution.md` | Infrastructure | Code runners |
| `04_implementation_guide/05_tool_implementations.md` | Application | 14 tools |
| `04_implementation_guide/06_agent_service_impl.md` | Application | ReAct loop |
| `04_implementation_guide/07_api_layer.md` | Presentation | FastAPI routes |
| `04_implementation_guide/08_testing.md` | — | Test strategy |
| `05_reference/01_configuration_reference.md` | — | All config options |
| `05_reference/02_tool_api_reference.md` | — | Tool quick-ref |
| `05_reference/03_troubleshooting.md` | — | Error guide |
| `05_reference/04_worked_example.md` | — | End-to-end example |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.12+ | `python --version` |
| uv | latest | `pip install uv` |
| Anthropic API key | — | [console.anthropic.com](https://console.anthropic.com) |
| Git | 2.x+ | for cloning |
| `jupyter_client` | 8.x+ | for Jupyter backend (optional) |
| `pint` | 0.24+ | for unit validation |
| `nbformat` | 5.x+ | for notebook export |
| `pandas` | 2.x+ | for dataset loading |

You do **not** need a local Jupyter server — the `JupyterKernelManager` backend launches a kernel in-process. All other code execution options work without any local Jupyter installation.

---

## Learning Path

This guide is designed to be read sequentially, but each section is self-contained:

```
New to AI agents?           → Start at 01_fundamentals/
Know the basics?            → Jump to 02_anthropic_sdk_reference/
Building right now?         → Use 04_implementation_guide/
Debugging an issue?         → See 05_reference/03_troubleshooting.md
Want the full picture fast? → Read 05_reference/04_worked_example.md
```

---

## License

MIT — see `LICENSE` file for details.
