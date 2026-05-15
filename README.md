![Banner](PSABaner.jpg)
# Procurement Asystent

> **An AI-powered multi-agent system for automating enterprise procurement workflows.**  
> Built with [LangGraph](https://github.com/langchain-ai/langgraph) and the [Anthropic Claude API](https://www.anthropic.com/).

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Agents](#agents)
  - [Intake Agent](#1-intake-agent)
  - [Procurement Agent](#2-procurement-agent)
  - [Analyst Agent](#3-analyst-agent)
  - [Orchestrator Agent](#4-orchestrator-agent)
- [Procurement Agent Tools](#procurement-agent-tools)
- [Procurement Process Types](#procurement-process-types)
- [Decision Outcomes](#decision-outcomes)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Requirements](#requirements)
- [Status](#status)

---

## Overview

**Procurement Asystent** is a multi-agent AI system designed to support enterprise procurement departments in automating the supplier search and purchase evaluation process.

The system processes incoming purchase requests for machinery, equipment, vehicles, services, and other goods — based on the company's defined purchasing policy configured in `enterprise_buying_rules.yaml`. It handles the full procurement intake cycle: from request validation and interactive clarification, through supplier search and cost analysis, to a final structured procurement decision with a plain-language summary for the requester.

This is an early-stage version of the system. Its scope is intentionally limited to the core procurement workflow.

---

## System Architecture

The system is built as a directed agent graph using **LangGraph**. Each agent runs as a separate node, receiving a shared state object (`SharedState`) and producing structured outputs that are passed downstream. Output schemas are enforced via **Pydantic**.

```
  User Input (CLI)
        │
        ▼
┌──────────────┐
│ Intake       │  ── Validates & classifies the request
│ Agent        │  ── Triggers interactive clarification rounds if required
│              │     fields are missing (GraphInterrupt → CLI → resume)
└──────┬───────┘
       │  validated_request, category_id, process_type
       ▼
┌──────────────┐
│ Procurement  │  ── Searches for suppliers (web search + PDF parsing)
│ Agent        │  ── Converts prices to a common currency
│              │  ── Builds purchasing strategy & negotiation points
└──────┬───────┘
       │  supplier_recommendations, procurement_strategy, negotiation_points
       ▼
┌──────────────┐
│ Analyst      │  ── Performs TCO analysis & risk assessment
│ Agent        │  ── Issues final decision (PROCEED / ESCALATE / ...)
└──────┬───────┘
       │  cost_analysis, risk_analysis, final_decision
       ▼
┌──────────────┐     ┌──────────────┐
│ Orchestrator │ ──► │ Human Review │  (if decision = ESCALATE)
│ Agent        │     │ Node         │
└──────┬───────┘     └──────────────┘
       │
       ▼
  Final Report + Message to User (CLI)
```

All agents share state through `SharedState`. The graph is built in `procurement_system/graph/procurement_graph.py`.

---

## Inter-Agent Communication Protocol

The multi‑agent procurement system uses a **state‑centric communication model** where agents never communicate directly. Instead, all coordination happens through a single, shared state object (`SharedState`) that flows through the LangGraph execution pipeline. This design ensures full auditability, loose coupling, and deterministic error recovery.

### Core Principles

| Principle | Implementation |
|-----------|----------------|
| **No direct agent‑to‑agent calls** | Agents read from and write to `SharedState` only. |
| **Write‑once, read‑many** | Each domain field is owned by exactly one agent (e.g. `intake.validated_request`), but can be read by any downstream agent. |
| **Decision‑driven routing** | Agents set `routing_decision` or `final_decision` fields. The graph’s conditional edges use these values to determine the next node. |
| **Append‑only audit trail** | Every agent appends to `decision_log` and `errors` using `Annotated[list[str], operator.add]`. No overwriting. |

### Shared State Structure

All agents receive the same `SharedState` TypedDict (defined in `procurement_system/state.py`):

```python
class SharedState(TypedDict, total=False):
    # Global fields (read/write by multiple agents)
    raw_request: str
    routing_decision: Literal["proceed", "escalate"]
    final_decision: Literal["PROCEED", "PROCEED_WITH_CONDITIONS", "ESCALATE", "REJECT"]
    decision_log: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # Domain‑specific sections (each owned by one agent)
    intake: IntakeState       # written by Intake Agent
    procurement: ProcurementState  # written by Procurement Agent
    analyst: AnalystState     # written by Analyst Agent
    orchestrator: OrchestratorState  # written by Orchestrator Agent
```

## Agents

### 1. Intake Agent

**File:** `procurement_system/agents/intake/agent.py`  
**Node:** `procurement_system/nodes/intake_node.py`  
**Prompt:** `procurement_system/prompts/intake_agent_system.txt`

The first agent in the workflow. It receives a raw user request and is responsible for:

- Parsing and validating required fields (`description`, `quantity`, `unit`)
- Classifying the request into a procurement category (defined in buying rules)
- Determining the appropriate process type
- Applying enterprise buying rules for routing decisions
- Triggering **interactive clarification rounds** when required fields are missing — the graph raises a `GraphInterrupt`, the CLI prompts the user, and the workflow resumes via `Command(resume=answer)`

Every routing decision references a specific buying rule for full audit traceability.

**Required input fields:**

| Field | Description |
|---|---|
| `description` | Full product or service description |
| `quantity` | Numeric quantity requested |
| `unit` | Unit of measure (e.g. `units`, `kg`, `hours`) |

**Optional input fields:** `estimated_unit_price_usd`, `deadline`, `urgency`, `requirements`

---

### 2. Procurement Agent

**File:** `procurement_system/agents/procurement/agent.py`  
**Node:** `procurement_system/nodes/procurement_node.py`  
**Prompt:** `procurement_system/prompts/procurement_agent_system.txt`

The second agent in the workflow. It receives the validated and classified request from the Intake Agent and is responsible for:

- Searching for **1–5 supplier options** from genuinely different market segments
- Defining a purchasing strategy narrative aligned with the process type
- Determining the recommended order type
- Preparing **3–5 specific, actionable negotiation points**
- Listing alternative products or services where applicable

This agent has access to **3 tools** — see [Procurement Agent Tools](#procurement-agent-tools).

Each supplier recommendation includes: name, type, estimated price range, lead time, reliability score, pros, cons, and contact priority.

---

### 3. Analyst Agent

**File:** `procurement_system/agents/analyst/agent.py`  
**Node:** `procurement_system/nodes/analyst_node.py`  
**Prompt:** `procurement_system/prompts/analyst_agent_system.txt`

The third agent in the workflow. It receives the validated request and supplier recommendations and is responsible for:

- Performing a **Total Cost of Ownership (TCO)** analysis
- Modelling **three cost scenarios**: optimistic, realistic, pessimistic
- Assessing **3–5 procurement risks** (financial, operational, supplier, quality, delivery, compliance)
- Calculating a composite `risk_score` using a deterministic formula
- Issuing a **final procurement decision** based on configurable thresholds

**Risk score formula:**
```
risk_score = average(probability × impact) across all identified risks
             clamped to range 1.0 – 10.0
```

**Decision thresholds** (configurable in `config/config_analyst_agent.yaml`):

| Threshold | Default |
|---|---|
| `auto_proceed` | ≤ 3.0 |
| `auto_escalate` | ≥ 7.5 |

---

### 4. Orchestrator Agent

**File:** `procurement_system/agents/orchestrator/agent.py`  
**Node:** `procurement_system/nodes/orchestrator_node.py`  
**Prompt:** `procurement_system/prompts/orchestrator_system.txt`

The final agent in the workflow. It receives the complete outputs of all three upstream agents and is responsible for:

- Compiling a structured **final report** (`LLMFinalReport` schema, enforced by Pydantic)
- Generating a **plain-language message** for the requester (max 150 words, no technical jargon)
- Preserving the full **decision log** for audit purposes
- Validating consistency of the final decision across all upstream outputs

If the Analyst Agent's decision is `ESCALATE`, the graph routes to the **Human Review Node** (`nodes/human_review_node.py`) before producing the final report.

The Orchestrator makes **no new procurement decisions** — it synthesises and communicates the decisions already made by upstream agents.

---

## Procurement Agent Tools

The Procurement Agent has access to three tools:

| Tool | File | Description |
|---|---|---|
| **Supplier Web Search** | `tools/supplier_web_search.py` | Searches the web for suppliers matching the request (powered by Tavily) |
| **PDF Reader** | `tools/pdf_reader.py` | Extracts and parses supplier documents, catalogues, and offers in PDF format |
| **Currency Converter** | `tools/currency_converter.py` | Converts supplier price quotes to a common currency for comparison |

Each tool has a corresponding service layer (`services/`) and repository layer (`repositories/`).

---

## Procurement Process Types

| Process Type | Description |
|---|---|
| `catalog_purchase` | Standard purchase from an approved catalogue |
| `rfq` | Request for Quotation — competitive quotes from suppliers |
| `formal_rfq` | Formal RFQ / RFP for higher-value purchases |
| `strategic_sourcing` | Long-term, high-value strategic procurement |

Process types and their value thresholds are defined in `config/enterprise_buying_rules.yaml`.

---

## Decision Outcomes

| Decision | Meaning |
|---|---|
| `PROCEED` | Purchase approved — risk and budget within thresholds |
| `PROCEED_WITH_CONDITIONS` | Approved with specific conditions to satisfy before signing |
| `ESCALATE` | Requires human review — risk or budget exceeds thresholds |
| `REJECT` | Request is fundamentally unfeasible |

---

## Project Structure

```
projekt/
├── .env                          # Local environment variables (not committed)
├── .env.example                  # Environment variable template
├── .gitignore
├── README.md
├── main.py                       # CLI entry point
├── requirements.txt
│
│
└── procurement_system/
    ├── constants.py
    ├── container.py              # Dependency injection container
    ├── exceptions.py             # Custom exceptions (e.g. StructuredOutputError)
    ├── project_paths.py
    ├── settings.py
    ├── state.py                  # SharedState definition
    │
    ├── agents/
    │   ├── base_agent.py
    │   ├── mixins.py
    │   ├── analyst/agent.py
    │   ├── intake/agent.py
    │   ├── orchestrator/agent.py
    │   └── procurement/agent.py
    │
    ├── config/
    │   ├── config_analyst_agent.yaml
    │   ├── config_intake_agent.yaml
    │   ├── config_orchestrator.yaml
    │   ├── config_procurement_agent.yaml
    │   ├── enterprise_buying_rules.yaml   # Core purchasing policy
    │   ├── logging.yaml
    │   └── model_registry.yaml
    │
    ├── graph/
    │   └── procurement_graph.py          # LangGraph graph definition
    │
    ├── nodes/
    │   ├── analyst_node.py
    │   ├── human_review_node.py          # Handles ESCALATE decisions
    │   ├── intake_node.py
    │   ├── orchestrator_node.py
    │   └── procurement_node.py
    │
    ├── prompts/
    │   ├── analyst_agent_system.txt
    │   ├── analyst_agent_user.txt
    │   ├── forced_summary_llm_tool_loop.txt
    │   ├── intake_agent_system.txt
    │   ├── intake_agent_user.txt
    │   ├── orchestrator_system.txt
    │   ├── orchestrator_user.txt
    │   ├── procurement_agent_system.txt
    │   └── procurement_agent_user.txt
    │
    ├── repositories/
    │   ├── currency_repository.py
    │   ├── pdf_repository.py
    │   └── tavily_repository.py
    │
    ├── schemas/
    │   ├── analyst_schemas.py
    │   ├── buying_rules_schemas.py
    │   ├── intake_schemas.py
    │   ├── orchestrator_schemas.py
    │   ├── procurement_schemas.py
    │   └── tool_schemas.py
    │
    ├── services/
    │   ├── currency_service.py
    │   ├── pdf_extraction_service.py
    │   └── supplier_web_search_service.py
    │
    ├── tools/
    │   ├── currency_converter.py
    │   ├── pdf_reader.py
    │   └── supplier_web_search.py
    │
    └── utils/
        ├── agent_llm_bundle.py
        ├── buying_rules_prompt_builder.py
        ├── llm.py
        ├── llm_router.py
        ├── logger.py
        ├── message_normalizer.py
        ├── model_resolver.py
        ├── prompt_assembler.py
        ├── prompt_loader.py        # Loads prompts with lru_cache
        ├── tool_utils.py
        └── yaml_loader.py

tests/
├── conftest.py
├── agents/
│   ├── test_analyst_agent.py
│   ├── test_intake_agent.py
│   ├── test_orchestrator_agent.py
│   └── test_procurement_agent.py
├── nodes/
│   ├── test_analyst_node.py
│   ├── test_intake_node.py
│   ├── test_orchestrator_node.py
│   └── test_procurement_node.py
├── repositories/
│   ├── test_currency_repository.py
│   ├── test_pdf_repository.py
│   └── test_tavily_repository.py
└── services/
    ├── test_currency_service.py
    ├── test_pdf_extraction_service.py
    └── test_supplier_web_search_service.py
```

---

## Installation

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Tavily API key](https://tavily.com/) (used by the Supplier Web Search tool)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/procurement-asystent.git
cd procurement-asystent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys
```

### Environment Variables (`.env`)

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## Configuration

### Enterprise Buying Rules

The core purchasing policy is defined in:

```
procurement_system/config/enterprise_buying_rules.yaml
```

This file defines procurement categories, value thresholds, process type assignments, framework agreements, and escalation rules. The Intake Agent reads and applies these rules to every incoming request.

### Agent Configuration

Each agent has its own YAML config file controlling model selection, prompt version, and agent-specific parameters.

**Example — Analyst Agent (`config/config_analyst_agent.yaml`):**

```yaml
version: "1.5.0"

decision_thresholds:
  auto_proceed: 3.0
  auto_escalate: 7.5
```

### Prompt Versioning

Any edit to a system prompt file requires:
1. Version bump in the corresponding config YAML
2. Updated SHA-256 checksum (`sha256sum <prompt_file>`)
3. Entry in the prompt file's `CHANGELOG` section

---

## Usage

Run the system from the command line:

```bash
python main.py
```

Enter a purchase requisition in natural language when prompted:

```
Provide purchase requisition (or exit): 2 units of industrial CNC milling machine, 3-axis, budget ~90000 USD, deadline September 2026
```

### Interactive Clarification

If the request is missing required fields, the system pauses and asks for them before continuing:

```
--- Question (round 1) ---
Missing fields: quantity, unit
Could you please provide the quantity and unit of measure for your request?
> 3 units
```

The workflow resumes automatically with the provided answer. Python controls the number of clarification rounds — the agent only formulates the question.

### Output

```
📄 RAPORT
============================================================
Decision: PROCEED_WITH_CONDITIONS
Message: Your request for 2 CNC milling machines has been reviewed...
--- Decision log ---
 • [intake] Request classified as MACHINERY / formal_rfq
 • [procurement] 3 supplier options identified
 • [analyst] risk_score: 3.8 — medium risk
 • [orchestrator] Final report compiled
```

Type `exit`, `quit`, or `q` to end the session.

---

## Running Tests

```bash
pytest tests/
```

Tests are organised by layer: agents, nodes, repositories, and services.

---

## Requirements

Key dependencies (see `requirements.txt` for pinned versions):

```
anthropic
langgraph
langchain
pydantic
python-dotenv
pyyaml
tavily-python
```

---

## Status

**Version:** 0.1.0 — Early development  
The core multi-agent workflow is functional via CLI. Features such as a web interface, persistent storage, and extended integrations are not yet implemented.
