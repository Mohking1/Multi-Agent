# WorkOS — Personal Executive AI Operating System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Native%20Subagent%20DAG-green.svg)]()
[![Memory](https://img.shields.io/badge/Memory-4--Network%20Loci--Hindsight-purple.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)]()

**WorkOS** is an autonomous personal executive AI operating system designed for knowledge workers, executives, and engineers. It replaces legacy, fragile multi-agent frameworks with a high-performance native asynchronous Python orchestration engine, an embedded 4-network cognitive memory system (Loci-Hindsight), autonomous email operations, Docling document intelligence, and Elasticsearch hybrid RAG retrieval.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       WORKOS EXECUTIVE PLANNER                          │
│  - Goal Decomposition & Intent Classification                           │
│  - Spatial Memory Context Injection (< 200 tokens wake-up cost)          │
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
│                 4-NETWORK COGNITIVE MEMORY (SQLITE FTS5)                │
│  • Facts (Verified)    • Experiences (Episodic)                         │
│  • Entities (Dossiers) • Beliefs (Superseding Preferences)              │
│  • Spatial Loci Routing: Wings (projects/people) / Halls (contacts/etc) │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

### 1. Hierarchical Executive Orchestration
* **Pure Native Async Python**: Zero reliance on AutoGen or external orchestrators.
* **Dynamic Plan DAGs**: Generates multi-step execution plans with dynamic variable interpolation (e.g. chaining email attachments directly into document parsers and knowledge indexing).
* **Grounded Synthesis**: Formulates user responses grounded strictly in verified subagent execution artifacts.
* **Async Memory Reflection**: Extracts facts, execution logs, and behavioral beliefs in the background without blocking interactive responses.

### 2. 4-Network Spatial Cognitive Memory (Loci-Hindsight)
* **Zero External SaaS Dependencies**: Stored in a fast, embedded SQLite database (`workos_memory.db`) with FTS5 BM25 full-text indexing.
* **4 Distinct Networks**:
  1. **Facts**: Atomic verified knowledge with confidence scores and timestamps.
  2. **Experiences**: Chronological execution logs and document processing history.
  3. **Entities**: Living dossiers of people, organizations, and accounts.
  4. **Beliefs**: Evolving user preferences with automated contradiction resolution (newer rules supersede older ones).
* **MemPalace Spatial Loci**: Hierarchical routing across Wings (`/projects/`, `/people/`, `/workflows/`, `/system/`) and Halls (`contacts`, `decisions`, `preferences`). Wake-up prompt injection requires **< 200 tokens**.

### 3. Specialist Subagents
* **Mail Specialist (`mail_agent`)**:
  * IMAP search (by sender, subject, date, attachment status) and thread extraction.
  * SMTP draft creation and message dispatch.
  * Mailbox organization (folder moves, flagging, read/unread status).
  * **Configurable Autonomy Engine**:
    * `SUPERVISED`: Sensitive actions (such as sending emails to non-whitelisted addresses) stage as drafts and require explicit confirmation.
    * `FULL`: Autonomous execution for whitelisted recipients.
* **Document Intelligence Specialist (`doc_agent`)**:
  * Powered by **Docling** for deep structure extraction across PDF, DOCX, PPTX, XLSX, and images.
  * **TableFormer** high-precision table extraction with cell coordinates.
  * Structure-aware semantic chunking preserving document hierarchies.
* **Knowledge & RAG Specialist (`rag_agent`)**:
  * **Elasticsearch 8 Hybrid Retrieval**: Reciprocal Rank Fusion (RRF) combining Lucene BM25 lexical search with dense vector embeddings.
  * **Parent-Child Chunk Resolution**: Ingests granular child passages linked to overarching parent sections for context-rich generation.
  * Multi-hop query routing and grounded citations.

### 4. Interactive Rich Console & CLI Entrypoint
* Live plan visualization with step-by-step progress spinners and status badges.
* Color-coded terminal UI with human-in-the-loop confirmation.
* Full suite of interactive slash commands.

---

## 💻 Interactive Console & Slash Commands

Launch the interactive executive console:

```bash
python main.py
```

### Slash Commands Reference

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `/help` | | Display all commands and interactive usage reference |
| `/policy` | `[FULL \| SUPERVISED]` | View or switch autonomy policy level |
| `/plan` | `<goal>` | Preview dynamic execution plan DAG without running it |
| `/memory` | `[summary \| loci]` | Display spatial Loci memory architecture summary |
| `/memory search` | `<query>` | Search cognitive memory with BM25 FTS5 ranking |
| `/memory beliefs` | `[wing]` | List active behavioral preferences and system rules |
| `/memory add` | `<wing> <hall> <key> <val>` | Retain a verified fact directly into memory |
| `/mail status` | | Check IMAP and SMTP configuration and connectivity |
| `/mail search` | `<query>` | Search mailbox for matching emails |
| `/mail draft` | `<to> <subject> <body>` | Stage an email draft in the mailbox |
| `/mail send` | `<to> <subject> <body>` | Send email (subject to active autonomy policy) |
| `/doc parse` | `<file_path>` | Parse document with Docling into structured Markdown |
| `/doc tables` | `<file_path>` | Extract structured tables from document |
| `/rag search` | `<query>` | Perform hybrid RRF search across Elasticsearch index |
| `/rag ingest` | `<file_path>` | Ingest PDF document into hybrid RAG index |
| `/rag ask` | `<query>` | Query RAG knowledge base with citation grounding |
| `/clear` | | Clear console screen |
| `/exit` | | Exit WorkOS console (`/quit`, `/q`) |

### Non-Interactive CLI Mode

Execute single goals or generate plans directly from the command line:

```bash
# Execute a multi-step goal in supervised mode
python main.py --goal "Search inbox for invoice from Arvind, parse PDF, and tell me the total"

# Generate and preview plan DAG without executing
python main.py --goal "Ingest quarterly financial report into RAG" --plan-only

# Run with full autonomy policy
python main.py --goal "Draft summary email to team" --autonomy FULL
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.12+
* Elasticsearch 8.x (optional for local RAG indexing)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Mohking1/Multi-Agent.git
cd Multi-Agent

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# LLM Provider
GEMINI_API_KEY=your_gemini_api_key
MODEL_NAME=gemini-3.7-flash

# Autonomy & Policies
AUTONOMY_LEVEL=SUPERVISED                 # "FULL" or "SUPERVISED"
AUTONOMOUS_TRUSTED_RECIPIENTS=vip@corp.com,team@corp.com

# Memory Database
WORKOS_MEMORY_DB=workos_memory.db

# Email Configuration (IMAP / SMTP)
IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_USER=your_email@gmail.com
IMAP_PASSWORD=your_app_password
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Elasticsearch Hybrid RAG
ELASTICSEARCH_URL=http://localhost:9200
RAG_INDEX_NAME=workos_knowledge_base
```

---

## 🧪 Running Tests

WorkOS includes comprehensive unit, integration, and smoke test suites:

```bash
# Run all tests
pytest -v

# Run CLI smoke tests
pytest tests/test_cli_smoke.py -v

# Run Memory Engine tests
pytest tests/test_memory_engine.py -v

# Run Executive Planner tests
pytest tests/test_planner.py -v
```

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
