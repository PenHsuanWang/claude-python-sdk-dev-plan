# Advanced Chapter — AI Data Analysis Agent with Code Execution & Jupyter

> **What Anthropic actually provides, what it doesn't, and how to build a production data analysis AI agent with interactive visualization.**

---

## The Short Answer

**Yes — but with important nuances about what "Jupyter" means in this context.**

Anthropic provides three different mechanisms relevant to your use case, each with distinct trade-offs:

| Mechanism | Who runs the code | Jupyter support | Interactive? |
|-----------|------------------|-----------------|-------------|
| [Code Execution Tool](#option-a-anthropics-code-execution-tool) | Anthropic's servers | ❌ No | ❌ Static charts only |
| [Custom Jupyter Kernel Bridge](#option-b-custom-jupyter-kernel-bridge) | Your server | ✅ Real kernel | ✅ Full interactivity |
| [Computer Use + Jupyter UI](#option-c-computer-use-tool--jupyter-desktop) | Your desktop/VM | ✅ Real Jupyter | ✅ True GUI interaction |

Choosing the right one depends on your production requirements. This chapter explains all three from first principles.

---

## What Anthropic Actually Provides (The Full Map)

As of 2026, Anthropic's Python SDK supports these **Anthropic-provided tools** relevant to data analysis:

```
Server Tools (Anthropic executes):          Client Tools (your code executes):
─────────────────────────────────           ──────────────────────────────────
code_execution_20260120  ← ★ key            bash_20250124
web_search_20260209                         text_editor_20250728
web_fetch_20260209                          computer_20251124  ← Jupyter GUI
                                            memory_20250818
```

The `code_execution_20260120` tool is the core of the data analysis story. Let's understand it deeply.

---

## Option A: Anthropic's Code Execution Tool

### What It Is

A **server-side sandboxed Python environment** that Anthropic runs for you. When you include this tool in your request, Claude can autonomously:

- Write and execute Python code
- Perform data analysis with pandas, numpy, scipy
- Generate charts with matplotlib, seaborn
- Read/write files (CSV, Excel, JSON, images)
- Run bash commands

**Pre-installed libraries (relevant subset):**

```
pandas       numpy       scipy       scikit-learn    statsmodels
matplotlib   seaborn     pillow      sympy           mpmath
pyarrow      openpyxl    xlsxwriter  xlrd            tabula-py
```

### Basic Data Analysis Agent

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=[
        {"type": "code_execution_20260120"}  # ← one line enables full Python env
    ],
    messages=[{
        "role": "user",
        "content": "Analyze this sales data and create a bar chart by region:\n\n"
                   "Region,Revenue,Units\nWest,4200000,1200\nEast,3800000,980\n"
                   "North,2900000,750\nSouth,3100000,820"
    }]
)
```

Claude will:
1. Write pandas code to parse the CSV
2. Generate a matplotlib chart
3. Return the chart as a base64-encoded image in the response

### Uploading Your Own Dataset (Files API)

For real datasets, upload files first and reference them by ID:

```python
import anthropic
from pathlib import Path

client = anthropic.Anthropic()

# Step 1: Upload the dataset once (store the file_id for reuse)
with open("sales_data.csv", "rb") as f:
    file_obj = client.beta.files.upload(
        file=("sales_data.csv", f, "text/plain"),
    )
file_id = file_obj.id
print(f"Uploaded → {file_id}")

# Step 2: Ask Claude to analyze it
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    betas=["files-api-2025-04-14"],         # ← required for Files API
    tools=[
        {"type": "code_execution_20260120"}
    ],
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Analyze this dataset. Show me: (1) summary statistics, "
                        "(2) a correlation heatmap, (3) top 5 anomalies by Z-score."
            },
            {
                "type": "container_upload",  # ← reference the uploaded file
                "file_id": file_id,
            }
        ]
    }]
)

# Step 3: Extract text + any generated chart images
for block in response.content:
    if hasattr(block, "text"):
        print(block.text)
    # Chart images are returned as image content blocks
    elif block.type == "image":
        # block.source.data is base64-encoded PNG
        import base64
        img_bytes = base64.b64decode(block.source.data)
        Path("chart.png").write_bytes(img_bytes)
        print("Chart saved → chart.png")
```

### Downloading Generated Files

After code execution generates files (charts, processed CSVs, Excel reports), retrieve them:

```python
# Find files created during execution (returned in response metadata)
container_id = response.model_fields_set  # check response for container info

# Use Files API to download
for block in response.content:
    if hasattr(block, "file_id"):
        file_content = client.beta.files.download(block.file_id)
        Path(f"output_{block.file_id}.png").write_bytes(file_content)
```

### Reusing Containers (State Persistence)

The code execution environment persists within a container (30-day lifetime, 4.5 min idle timeout). Reuse it for multi-turn analysis sessions:

```python
first_response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=[{"type": "code_execution_20260120"}],
    messages=[{"role": "user", "content": "Load sales.csv and compute monthly totals."}]
)

# Extract container ID from the first response
container_id = first_response.container.id   # e.g., "cntr_01ABC..."

# Second request reuses the same Python environment
#   — the DataFrame from the first request is still in memory
second_response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=[{"type": "code_execution_20260120"}],
    container={"id": container_id},           # ← reuse state
    messages=[
        # ... include full conversation history ...
        {"role": "user", "content": "Now plot those monthly totals as a line chart."}
    ]
)
```

### ⚠️ Limitations of the Code Execution Tool

| Limitation | Detail |
|---|---|
| **No internet access** | Completely isolated — cannot `pip install` or fetch external data |
| **No Jupyter kernel** | Outputs are images/text, not interactive notebook cells |
| **No ipywidgets** | No sliders, dropdowns, or interactive controls |
| **No plotly/dash** | Only matplotlib/seaborn (static charts) |
| **1 CPU, 5 GiB RAM** | Not suitable for large-scale ML training |
| **Not ZDR-eligible** | Data retained per Anthropic's standard policy |

---

## Option B: Custom Jupyter Kernel Bridge

This is the **recommended approach** for true Jupyter interactivity. You run a real Jupyter kernel on your own server and expose it to Claude as a custom tool.

### Architecture

```
User ──▶ FastAPI ──▶ AgentService ──▶ Claude API (with custom tools)
                                              │
                          stop_reason == "tool_use"
                                              │
                          execute_jupyter_cell(code)
                                              │
                     ┌────────────────────────▼───────────────────────┐
                     │        Your Jupyter Kernel (subprocess)         │
                     │   jupyter_client → IPython kernel               │
                     │   Returns: stdout, images (base64), errors      │
                     └──────────────────────────────────────────────── ┘
                                              │
                          tool_result → Claude
                                              │
                          stop_reason == "end_turn" → User sees answer
                                              │
                              Optionally: export session as .ipynb
```

### Implementation

**Install dependencies:**

```bash
uv add jupyter-client ipykernel nbformat plotly ipywidgets
```

**The Jupyter Kernel Tool:**

```python
# app/services/jupyter_tool.py
"""Jupyter kernel bridge — executes code in a persistent IPython kernel."""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import jupyter_client

logger = logging.getLogger(__name__)


class JupyterKernelManager:
    """Manages a single persistent Jupyter kernel for one analysis session."""

    def __init__(self):
        self._km: jupyter_client.KernelManager | None = None
        self._kc: jupyter_client.KernelClient | None = None
        self._cell_outputs: list[dict] = []   # for .ipynb export

    async def start(self):
        """Start a new IPython kernel."""
        self._km = jupyter_client.KernelManager(kernel_name="python3")
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        await asyncio.sleep(1.5)  # wait for kernel to initialize
        logger.info("Jupyter kernel started: %s", self._km.kernel_id)

    def execute_cell(self, code: str, timeout: int = 60) -> dict[str, Any]:
        """
        Execute a code cell in the kernel.
        Returns a dict with: stdout, stderr, images (base64), error.
        """
        if self._kc is None:
            return {"error": "Kernel not started. Call start() first."}

        msg_id = self._kc.execute(code)
        outputs = []
        images = []
        stderr_lines = []
        error_text = None

        try:
            while True:
                reply = self._kc.get_iopub_msg(timeout=timeout)
                msg_type = reply["msg_type"]
                content = reply["content"]

                if msg_type == "stream":
                    if content["name"] == "stdout":
                        outputs.append(content["text"])
                    elif content["name"] == "stderr":
                        stderr_lines.append(content["text"])

                elif msg_type == "execute_result":
                    text = content["data"].get("text/plain", "")
                    outputs.append(text)

                elif msg_type == "display_data" or msg_type == "update_display_data":
                    # Capture matplotlib/plotly charts as PNG
                    if "image/png" in content["data"]:
                        images.append(content["data"]["image/png"])  # already base64

                elif msg_type == "error":
                    error_text = f"{content['ename']}: {content['evalue']}\n"
                    error_text += "\n".join(content["traceback"])

                elif msg_type == "status" and content["execution_state"] == "idle":
                    break  # cell finished

        except Exception as exc:
            return {"error": f"Kernel communication error: {exc}"}

        # Record for .ipynb export
        self._cell_outputs.append({
            "code": code,
            "stdout": "".join(outputs),
            "images": images,
            "error": error_text,
        })

        result = {
            "stdout": "".join(outputs) or "(no output)",
            "stderr": "".join(stderr_lines) or None,
            "images": images,  # list of base64 PNG strings
            "error": error_text,
        }
        logger.debug("Cell executed, outputs=%d, images=%d", len(outputs), len(images))
        return result

    def export_notebook(self, path: str = "session.ipynb") -> str:
        """Export the session as a Jupyter notebook (.ipynb)."""
        import nbformat

        nb = nbformat.v4.new_notebook()
        cells = []
        for item in self._cell_outputs:
            cell_outputs = []

            if item["stdout"] and item["stdout"] != "(no output)":
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=item["stdout"]
                ))

            for img_b64 in item["images"]:
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="display_data",
                    data={"image/png": img_b64}
                ))

            if item["error"]:
                cell_outputs.append(nbformat.v4.new_output(
                    output_type="error",
                    ename="Error",
                    evalue=item["error"],
                    traceback=[]
                ))

            cells.append(nbformat.v4.new_code_cell(
                source=item["code"],
                outputs=cell_outputs
            ))

        nb.cells = cells
        nbformat.write(nb, path)
        logger.info("Notebook exported → %s", path)
        return path

    async def shutdown(self):
        if self._km:
            self._km.shutdown_kernel()
            logger.info("Jupyter kernel shut down.")


# Global manager (one per process for MVP; use session-scoped for production)
kernel_manager = JupyterKernelManager()
```

**Tool definitions for Claude:**

```python
# app/services/jupyter_tool_definitions.py
from typing import Any

JUPYTER_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "execute_python_cell",
        "description": (
            "Execute a block of Python code in a persistent Jupyter kernel. "
            "The kernel maintains state between calls — variables, imports, and "
            "DataFrames defined in previous cells remain available. "
            "Use this for: data loading, cleaning, EDA, statistical analysis, "
            "creating visualizations (matplotlib, plotly, seaborn). "
            "Returns stdout text output and any generated chart images. "
            "Always write complete, runnable Python — do not use magic commands (%)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Python code to execute. Include all necessary imports. "
                        "For charts, call plt.show() or fig.show() at the end of the cell."
                    )
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "export_notebook",
        "description": (
            "Export the entire analysis session as a Jupyter notebook (.ipynb file). "
            "Call this at the end of the analysis to save all code and outputs. "
            "The notebook can be opened in JupyterLab or VS Code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Output filename (e.g., 'sales_analysis.ipynb')"
                }
            },
            "required": ["filename"]
        }
    }
]
```

**Tool executor:**

```python
# app/services/jupyter_executor.py
import base64
import json
from app.services.jupyter_tool import kernel_manager


def execute_python_cell(code: str) -> str:
    """Run code in the Jupyter kernel and return a descriptive string for Claude."""
    result = kernel_manager.execute_cell(code)

    parts = []

    if result.get("error"):
        parts.append(f"ERROR:\n{result['error']}")
    else:
        if result.get("stdout") and result["stdout"] != "(no output)":
            parts.append(f"OUTPUT:\n{result['stdout']}")
        if result.get("images"):
            parts.append(f"CHARTS GENERATED: {len(result['images'])} image(s) rendered.")

    return "\n".join(parts) if parts else "(Cell executed successfully, no output)"


def execute_notebook_export(filename: str) -> str:
    path = kernel_manager.export_notebook(filename)
    return f"Notebook exported successfully → {path}"


JUPYTER_TOOL_REGISTRY = {
    "execute_python_cell": lambda inp: execute_python_cell(inp["code"]),
    "export_notebook": lambda inp: execute_notebook_export(inp.get("filename", "session.ipynb")),
}
```

**The data analysis agent service:**

```python
# app/services/data_analysis_agent.py
"""AgentService variant specialized for data analysis with Jupyter kernel."""

import asyncio
import logging
from typing import Any

from anthropic import AsyncAnthropic
from app.core.config import settings
from app.domain.models import AgentSession
from app.services.jupyter_tool import kernel_manager
from app.services.jupyter_executor import JUPYTER_TOOL_REGISTRY
from app.services.jupyter_tool_definitions import JUPYTER_TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

_client = AsyncAnthropic(
    api_key=settings.anthropic_api_key.get_secret_value(),
    max_retries=settings.max_retries,
)

_MAX_LOOP_ITERATIONS = 20  # data analysis may need more iterations
_SYSTEM_PROMPT = """You are an expert data scientist and Python programmer.
When analyzing data:
1. ALWAYS start by loading and inspecting the data (head(), info(), describe())
2. Clean and validate data before analysis
3. Create clear, well-labeled visualizations using matplotlib or plotly
4. Explain your findings in plain language after each analysis step
5. If an execution error occurs, diagnose the problem and fix the code
6. At the end of a complete analysis, export the session as a notebook

You have access to a persistent Jupyter Python kernel with pandas, numpy,
matplotlib, seaborn, plotly, scipy, and scikit-learn pre-available."""


class DataAnalysisAgentService:
    """Orchestrates Claude + Jupyter kernel for interactive data analysis."""

    async def start_kernel(self):
        await kernel_manager.start()

    async def run(self, session: AgentSession) -> str:
        for iteration in range(_MAX_LOOP_ITERATIONS):
            logger.debug("Data analysis loop iteration %d", iteration)

            response = await _client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.max_tokens,
                system=_SYSTEM_PROMPT,
                tools=JUPYTER_TOOL_DEFINITIONS,
                messages=session.messages,
            )

            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                session.add_assistant_message([
                    b.model_dump() if hasattr(b, "model_dump") else b
                    for b in response.content
                ])
                return final_text

            if response.stop_reason == "tool_use":
                session.add_assistant_message([
                    b.model_dump() if hasattr(b, "model_dump") else b
                    for b in response.content
                ])

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info("Jupyter tool: %s", block.name)
                        handler = JUPYTER_TOOL_REGISTRY.get(block.name)
                        if handler:
                            result = handler(dict(block.input))
                        else:
                            result = f"Error: Unknown tool '{block.name}'"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                session.add_tool_results(tool_results)
                continue

            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason}")

        raise RuntimeError(f"Analysis loop exceeded {_MAX_LOOP_ITERATIONS} iterations.")

    async def get_latest_charts(self) -> list[bytes]:
        """Return the latest generated chart images as PNG bytes."""
        return [
            __import__("base64").b64decode(img)
            for entry in kernel_manager._cell_outputs
            for img in entry.get("images", [])
        ]

    async def shutdown(self):
        await kernel_manager.shutdown()


data_analysis_agent = DataAnalysisAgentService()
```

### Using It End-to-End

```python
# Example: complete data analysis session
import asyncio
from app.services.data_analysis_agent import data_analysis_agent
from app.domain.models import AgentSession

async def main():
    await data_analysis_agent.start_kernel()
    
    session = AgentSession(session_id="analysis-01")
    session.add_user_message(
        "I have sales data in sales.csv. Please:\n"
        "1. Load and inspect it\n"
        "2. Show monthly revenue trends (line chart)\n"
        "3. Find the top 3 performing regions\n"
        "4. Identify any anomalies\n"
        "5. Export the full analysis as a notebook"
    )
    
    answer = await data_analysis_agent.run(session)
    print("Analysis complete:\n", answer)
    
    charts = await data_analysis_agent.get_latest_charts()
    for i, chart in enumerate(charts):
        open(f"chart_{i}.png", "wb").write(chart)
        print(f"Saved chart_{i}.png")
    
    await data_analysis_agent.shutdown()

asyncio.run(main())
```

---

## Option C: Computer Use Tool + Jupyter Desktop

For **true GUI interaction** — Claude opens a real browser, navigates to Jupyter, clicks cells, and interacts with plotly widgets — use Anthropic's Computer Use tool.

### What Computer Use Provides

- **Screenshot capture** — Claude sees the current screen state
- **Mouse control** — click, drag, scroll
- **Keyboard input** — type code, press Enter
- **Desktop automation** — interact with any running application

### Setup (Docker-based)

```bash
# Clone the Anthropic reference implementation
git clone https://github.com/anthropics/anthropic-quickstarts
cd anthropic-quickstarts/computer-use-demo

# Start the full desktop environment (X11 + Jupyter + browser)
docker build -t claude-computer-use .
docker run -p 6080:6080 -p 8888:8888 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  claude-computer-use
```

### The Computer Use Agent

```python
import anthropic

client = anthropic.Anthropic()

# Computer use requires a beta header
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    betas=["computer-use-2025-11-24"],    # ← required beta header
    tools=[
        {
            "type": "computer_20251124",  # ← Anthropic-provided schema
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080,
        }
    ],
    messages=[{
        "role": "user",
        "content": (
            "Open Jupyter notebook in the browser, create a new Python notebook, "
            "load the file /data/sales.csv into a pandas DataFrame, and create "
            "an interactive plotly chart showing revenue by region."
        )
    }]
)

# Handle the tool_use loop — each iteration:
# 1. Claude returns a tool_use block with an action (screenshot, click, type)
# 2. Your code executes that action on the actual desktop
# 3. You capture a screenshot and return it as the tool_result
# 4. Claude sees the screenshot and decides the next action
```

### When to Use Computer Use

✅ **Good for:**
- True interactive Jupyter widgets (ipywidgets, plotly dash)
- Workflows that require clicking UI elements (not just code)
- Demos where visual proof of notebook execution matters

❌ **Not good for:**
- Production batch processing (too slow, ~2-5 sec per action)
- High-concurrency APIs (requires a desktop VM per session)
- Automated pipelines without a human in the loop

---

## Decision Framework: Which Option Should You Use?

```
Do you need true interactive widgets (sliders, dropdowns)?
├─ YES → Option C (Computer Use + Jupyter GUI)
│         or Option B (Custom Kernel Bridge + streamed images)
└─ NO
    ├─ Do you need plotly/dash interactive charts?
    │   └─ YES → Option B (Jupyter Bridge with plotly)
    └─ NO (static matplotlib/seaborn charts are fine)
        ├─ Do you want zero infrastructure setup?
        │   └─ YES → Option A (Anthropic Code Execution Tool)
        └─ NO (you have your own server)
            └─ Option B (Jupyter Bridge, more control)
```

### Feature Comparison

| Feature | Option A (Code Exec) | Option B (Jupyter Bridge) | Option C (Computer Use) |
|---------|---------------------|--------------------------|------------------------|
| Setup complexity | ⭐ Minimal | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Complex |
| Infrastructure cost | Anthropic's servers | Your server | Your VM/Docker |
| matplotlib/seaborn | ✅ Yes | ✅ Yes | ✅ Yes |
| plotly interactive | ❌ No | ✅ Yes | ✅ Yes |
| ipywidgets | ❌ No | ✅ Yes | ✅ Yes |
| Export to .ipynb | ❌ No | ✅ Yes | ✅ Yes (via screenshot) |
| pip install custom libs | ❌ No | ✅ Yes | ✅ Yes |
| Multi-user concurrent | ✅ Yes | ⚠️ Kernel-per-session | ❌ Desktop-per-session |
| Data privacy (ZDR) | ❌ Not eligible | ✅ Your infrastructure | ✅ Your infrastructure |
| Speed | Fast | Fast | Slow (GUI actions) |

---

## Programmatic Tool Calling (Advanced Pattern)

The latest `code_execution_20260120` adds a critical new capability: **Claude can call your custom tools from inside its Python code**, not just via the API loop. This is Anthropic's most powerful pattern for data analysis pipelines.

```python
# Claude writes Python like this inside code execution:
#   results = []
#   for region in ["West", "East", "North", "South"]:
#       data = await query_database(f"SELECT * FROM sales WHERE region='{region}'")
#       results.append(data)
#   df = pd.concat(results)
#   df.plot(kind="bar")

client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    tools=[
        {"type": "code_execution_20260120"},  # enables REPL + programmatic calling
        {
            "name": "query_database",
            "description": "Query the sales database and return a JSON array of records.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["sql"]
            },
            "allowed_callers": ["code_execution_20260120"],  # ← callable from inside Python
        }
    ],
    messages=[{"role": "user", "content": "Analyze sales by region and create visualizations."}]
)
```

**Why this matters for data analysis:**
- Claude can write a loop that queries 50 regions without 50 separate API round-trips
- Large query results are filtered inside the sandbox before entering Claude's context window
- Dramatically reduces token usage and latency for large datasets

---

## Summary

| What you want | Use |
|---|---|
| Quick data analysis, no setup | `code_execution_20260120` (server tool) |
| Upload your CSV/Excel, get charts back | Code Execution + Files API |
| True Jupyter kernel with state | Custom `JupyterKernelManager` bridge (Option B) |
| Export real `.ipynb` notebooks | Option B + `nbformat` |
| Interactive plotly/ipywidgets | Option B or Computer Use (Option C) |
| Claude controls Jupyter UI | Computer Use Tool (beta, Docker required) |
| Query DB in a loop efficiently | Programmatic Tool Calling (`allowed_callers`) |

**The fundamental answer:** Anthropic does not provide native Jupyter notebook integration — but it provides the primitives (Code Execution Tool, Files API, Computer Use, Programmatic Tool Calling) to build every level of that experience yourself, from zero-setup chart generation to full interactive Jupyter automation.
