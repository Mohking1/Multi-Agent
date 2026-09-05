# WorkOS — Autonomous Executive AI Operating System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Inference](https://img.shields.io/badge/Inference-100%25%20Local%20Ollama-emerald.svg)](https://ollama.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Native%20Subagent%20DAG-green.svg)]()
[![Memory](https://img.shields.io/badge/Memory-4--Network%20Loci--Hindsight-purple.svg)]()
[![UI](https://img.shields.io/badge/UI-Minimalist%20Web%20Dashboard-cyan.svg)]()
[![Code Quality](https://img.shields.io/badge/Linter-Ruff-orange.svg)](https://github.com/astral-sh/ruff)

**WorkOS** is an autonomous executive AI operating system designed for knowledge workers, executives, and engineers worldwide. It runs **100% locally and offline** via **Ollama**, featuring a native subagent execution DAG, an embedded 4-network cognitive memory engine (Loci-Hindsight), lossless original Document Vault with side-by-side citation grounding, autonomous email operations, Docling document intelligence, and Elasticsearch hybrid RAG.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       WORKOS EXECUTIVE PLANNER                          │
│  - 100% Local Inference via Ollama (JSON Schema Structured Planning)    │
│  - Goal Decomposition & Intent Classification                           │
│  - Spatial Memory Context Injection (< 150 tokens wake-up cost)          │
│  - Dynamic Execution Plan Formulation (JSON Schema DAG)                 │
│  - Variable Interpolation ($step_1.saved_path -> $step_2.file_path)     │
│  - Grounded Output Synthesis & Post-Execution Async Reflection          │
└───────┬─────────────────────────┬─────────────────────────┬─────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│  MAIL AGENT  │          │  DOC AGENT   │          │  RAG AGENT   │
│  IMAP / SMTP │          │   Docling    │          │ Elasticsearch│
│   Organize   │          │ TableFormer  │          │  Hybrid RRF  │
│ Autonomy Pol │          │   Chunker    │          │ Parent-Child │
└──────────────┘          └──────────────┘          └──────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 DOCUMENT VAULT & CITATION PROVENANCE                    │
│  • Permanent lossless storage of original source documents (PDF/DOCX)   │
│  • SHA-256 deduplication and page-level citation references             │
│  • Side-by-side original viewer + extracted Docling markdown            │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 4-NETWORK COGNITIVE MEMORY (SQLITE FTS5)                │
│  • Facts (Verified)    • Experiences (Episodic)                         │
│  • Entities (Dossiers) • Beliefs (Superseding Preferences)              │
│  • Spatial Loci Routing: Wings (projects/people) / Halls (contacts/etc) │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

### 1. Minimalist Aesthetic Web UI
* **Clean Nordic Dark Interface**: Deep slate/obsidian palette (`#0B0F17`) with Tailwind CSS, micro-interactions, and Alpine.js.
* **Executive Workspace**: Live DAG execution visualizer, real-time agent status tracking, and grounded synthesis with clickable citation chips.
* **Document Vault & Citation Inspector**: Upload, browse, and preview original stored documents alongside extracted tables and Markdown.
* **Cognitive Memory Browser**: Live 4-network inspector, Spatial Loci tree map, and FTS5 BM25 search.
* **Autonomous Mail Hub**: Inbox triage, policy status pill (`FULL` vs `SUPERVISED`), and staged draft approvals.

### 2. Document Vault & Audit-Grade Provenance
* **Lossless Storage**: Automatically preserves original uploaded PDFs, DOCXs, and email attachments in `data/vault/`.
* **SHA-256 Deduplication**: Prevents duplicate storage while maintaining metadata and parsed markdown history.
* **Interactive Citations**: RAG outputs provide structured citation chips `[1]`, `[2]` linking directly to the original stored document, snippet, and page number.

### 3. 100% Local Inference with Ollama
* Native `OllamaClient` interfacing directly with `http://localhost:11434`.
* Dynamic plan generation using JSON schema constraint (`format="json"`).
* Zero cloud API keys or external SaaS telemetry required.
* Hot-swap models on the fly via Web UI or CLI (`/models`, `/model <name>`).

### 4. 4-Network Spatial Cognitive Memory (Loci-Hindsight)
* Fast embedded SQLite storage with FTS5 BM25 search.
* 4 distinct networks: **Facts**, **Experiences**, **Entities**, and **Beliefs** (with automatic rule superseding).
* Spatial Loci wake-up cost: **< 150 tokens**.

---

## 🚀 Quickstart & Usage

### 1. Start the Minimalist Web UI
```bash
# Launch the web interface (opens on http://localhost:8000)
python ui.py

# Or via CLI flag
python main.py --ui --port 8000
```

### 2. Interactive CLI Console
```bash
python main.py
```

### 3. Direct Headless Goal Execution
```bash
python main.py --goal "Search my emails for recent invoices, parse line items, and index into RAG"
```

### 4. Code Formatting & Linting (Ruff)
```bash
# Run lint checks with auto-fixes
ruff check --fix .

# Format all code
ruff format .
```

---

## 💻 CLI Slash Commands Reference

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `/help` | | Display all commands and interactive usage reference |
| `/ui` | | Information and status on Web UI dashboard |
| `/policy` | `[FULL \| SUPERVISED]` | View or switch autonomy policy level |
| `/models` | | List all models installed in local Ollama instance |
| `/model` | `<model_name>` | Switch active generative model on the fly |
| `/plan` | `<goal>` | Preview dynamic execution plan DAG without running it |
| `/memory` | `[summary \| loci]` | Display spatial Loci memory architecture summary |
| `/memory search` | `<query>` | Search cognitive memory with BM25 full-text ranking |
| `/doc parse` | `<path>` | Parse document into Markdown using Docling |
| `/doc tables` | `<path>` | Extract structured tables using TableFormer |
| `/rag ask` | `<question>` | Query RAG knowledge base with grounded citations |
| `/mail search` | `<query>` | Search inbox emails via IMAP |
| `/exit` | | Safely exit interactive console |

