# WorkOS: Autonomous Executive AI Operating System

WorkOS is an offline-first executive AI operating system that translates natural language goals into deterministic multi-agent execution workflows. It coordinates specialized subagents across email management, multimodal document intelligence, parent-child hybrid retrieval-augmented generation, and live web research.

The platform is designed to run locally using compact tool-calling language models (such as 3B parameter models in Ollama) while supporting optional cloud fallbacks such as Google Gemini. WorkOS decouples macro-planning from tool execution: an executive planner compiles goals into a directed acyclic graph (DAG), executes steps in topological order with typed variable resolution, passes state across a shared blackboard, confines subagents to isolated micro-ReAct loops, and preserves factual grounding through persistent cognitive memory and a lossless document vault.

---

## Execution Engine and DAG Orchestration

### Planner and Topological Graph Execution

WorkOS enforces strict separation between planning and execution. The Executive Planner does not execute tools directly; it decomposes user intent into an `ExecutionPlan` containing an array of `PlanStep` definitions. Each step specifies:

- An assigned specialist agent (`mail_agent`, `doc_agent`, `rag_agent`, or `web_agent`).
- A concise description of the step objective.
- An input data dictionary containing parameters or variable references.
- An explicit dependency list (`dependencies: [step_id, ...]`).

The engine resolves execution order using `graphlib.TopologicalSorter`. Steps with satisfied dependencies run sequentially or concurrently, and cyclical dependencies raise an explicit `CycleError`.

### Shared Blackboard and Variable Resolution

Inter-step communication is managed through a typed `BlackboardState`:

- **Variable Interpolation**: Step inputs can reference outputs from prerequisite steps using string tokens (such as `$step_1.saved_paths`, `$step_1.artifacts`, `$step_1.data.key`, or `$prev.data`) or typed `StepReference(step_id, output_key)` objects. The engine recursively resolves variables across dictionaries, lists, and nested objects before dispatching a task.
- **Execution Receipts**: Upon step completion, specialist agents emit compact execution receipts summarizing actions and extracted values. These receipts are appended to the blackboard, keeping upstream context compact without exhausting context windows.
- **Artifact Registry**: Output file paths, downloaded attachments, and extracted figures are tracked across steps and registered into the system vault for downstream consumption.

### Isolated Subagent Micro-ReAct Runtimes

Local 3B parameter models experience degraded accuracy and tool hallucination when exposed to large numbers of tool definitions simultaneously. WorkOS mitigates this by scoping each specialist subagent (`BaseSubagent`) to an isolated runtime:

- Subagents receive only their own domain-specific tool schemas (typically 3 to 6 tools).
- The agent executes a structured Thought, Action, Observation reasoning cycle using valid JSON responses:
  - Tool invocation format: specifies tool name and typed arguments.
  - Completion format: provides final findings and extracted structured data.
- **Direct Dispatch Bypass**: If a step instruction matches a registered tool name directly, the subagent bypasses the LLM reasoning hop and executes the tool immediately, minimizing latency and token overhead.

### Autonomy Governance and Safety Gates

Mutating and outbound actions are governed by an `AutonomyLevel` policy:

- **SUPERVISED** (Default): Outbound communications (such as sending emails to recipients outside the configured trusted whitelist) are intercepted. The action is redirected to stage an email in the IMAP `Drafts` folder, returning a pending confirmation status rather than transmitting over SMTP.
- **FULL**: Outbound actions execute autonomously according to plan specifications.
- **Trusted Recipients**: A comma-separated whitelist (`AUTONOMOUS_TRUSTED_RECIPIENTS`) exempts verified contacts from supervised approval gates.

---

## Specialist Subagents and Toolkits

### Mail Agent (`mail_agent`)

The Mail Agent manages IMAP and SMTP operations, message classification, and inbox organization.

- **IMAP Search and Retrieval**: Filters mailbox messages by sender, subject keyword, temporal ranges (`date_gte`, `date_lt`), read status (`seen`), and body text. Fetches full MIME structures, headers, plain text, and HTML bodies by UID.
- **Attachment Extraction**: Downloads attachments matching specified criteria, saving them to disk and passing file paths to downstream steps for parsing or ingestion.
- **Policy-Gated Email Dispatch**: Evaluates outbound email requests against autonomy rules and the trusted recipient whitelist, dispatching via SMTP when permitted or staging to the `Drafts` folder when supervised.
- **Autonomous Mailbox Organization**:
  - LLM-assisted specification extraction: parses high-level sorting goals into target folder names, subfolder rules (such as partitioning by sender domain or date), search terms, and temporal boundaries.
  - Search hypothesis generation: combines known cognitive entities and domain acronyms with search keywords.
  - Folder hierarchy management: inspects IMAP folder hierarchies, creates missing mailboxes using server-specific delimiters, and moves matching messages.

Key tools:
- `search_emails`: Query mailbox messages using structured criteria.
- `fetch_email`: Retrieve complete email content and metadata by UID.
- `download_attachment`: Download a single attachment by filename.
- `download_attachments`: Download all attachments matching an optional pattern.
- `create_draft`: Stage an email draft in the mailbox without sending.
- `send_email`: Send an email via SMTP (gated by autonomy policy).
- `move_email`: Move an email to a destination folder.
- `mark_email`: Update read (`seen`) or starred (`flagged`) message flags.
- `list_folders`: Inspect mailbox folders and hierarchy delimiters.
- `create_folder`: Create a new mailbox folder.
- `organize_emails`: Bulk-categorize and relocate messages based on criteria.

### Document Agent (`doc_agent`)

The Document Agent provides document parsing, tabular data extraction, and semantic chunking using Docling, IBM TableFormer, and HybridChunker.

- **Multimodal Document Parsing**: Converts PDF, DOCX, PPTX, HTML, Markdown, and image files into structured Markdown, JSON, plain text, or doctags with optional OCR.
- **TableFormer Extraction**: Executes deep table structure recognition (`TableFormerMode.ACCURATE`) to extract complex tables, retaining cell spans, column hierarchies, and data alignments as structured dataframes and clean Markdown tables.
- **Visual Asset Extraction**: Extracts embedded figures, charts, and diagrams, exporting them to disk with artifact tracking.
- **Structure-Aware Chunking**: Chunks documents along semantic and hierarchical boundaries (headings, sections, tables) using `HybridChunker` rather than fixed token lengths, preserving tabular context and heading relationships.
- **Deterministic Document Identification**: Generates reproducible document identifiers (`doc_<clean_name>_<sha256_prefix>`) for deduplication and session caching.

Key tools:
- `parse_document`: Convert a document into structured format with optional OCR and TableFormer support.
- `extract_tables`: Extract structured tables with cell matrices, headers, and dimensions.
- `extract_figures`: Extract embedded figures, charts, and images to disk.
- `structure_aware_chunk`: Segment a parsed document into hierarchical chunks for indexing.
- `get_stored_document`: Retrieve previously parsed document structure from cache.
- `list_stored_documents`: List cached documents and parsing metadata.

### RAG Agent (`rag_agent`)

The RAG Agent provides parent-child hybrid retrieval across an Elasticsearch knowledge base, combining lexical matching and dense vector embeddings.

- **Parent-Child Index Architecture**: Documents are indexed across two linked Elasticsearch indices:
  - Child index: contains granular chunks with dense vector embeddings (`bge-m3:latest` or deterministic high-dimensional projections) and BM25 text fields for precise retrieval.
  - Parent index: contains complete sections and context blocks. Matching child chunks resolve their parent sections at query time, providing full contextual breadth to the generation model.
- **Reciprocal Rank Fusion (RRF)**: Merges BM25 lexical ranking scores and dense vector cosine similarity scores, balancing keyword precision with semantic retrieval.
- **Mathematical Query Complexity Routing**: Evaluates incoming queries using an algorithmic classifier:
  - Shannon entropy: measures token distribution uncertainty across query terms.
  - IDF specificity proxy: computes the ratio of domain-specific rare tokens to common stop words.
  - Token density and reasoning indicators: detects comparative phrasing, question words, and multi-condition requirements.
  - Routes queries to `DIRECT_SEARCH` (low complexity, high specificity), `HYBRID_RAG` (balanced retrieval), or `MULTI_HOP_REASONING` (high entropy, complex synthesis).
- **Autonomous Multi-Phase Search**: Executes query expansion (generating 2 to 3 semantic variations), candidate retrieval across variations, LLM precision re-ranking of chunk snippets, and grounded synthesis with page-level citations.

Key tools:
- `rag_search`: Hybrid search combining BM25 and dense vectors with RRF fusion and parent context retrieval.
- `rag_ingest_pdf`: Ingest a document or PDF into parent-child Elasticsearch indices.
- `rag_ask`: Search the knowledge base and synthesize a grounded answer with citations.
- `rag_route_query`: Classify query complexity to select the optimal retrieval strategy.

### Web Intelligence Agent (`web_agent`)

The Web Agent provides real-time internet intelligence without dependencies on proprietary cloud search APIs.

- **Zero-Cloud Metasearch**: Connects to a local SearXNG metasearch instance (`http://localhost:8080`) aggregating results from Google, Bing, DuckDuckGo, and Brave. Automatically falls back to direct DuckDuckGo HTML scraping if SearXNG is unavailable.
- **Deep Markdown Extraction**: Uses Trafilatura to fetch target web pages and extract primary body text into clean Markdown, removing navigation headers, footers, advertisements, and tracking scripts.
- **Four-Phase Autonomous Research**:
  1. Hypothesis Generation: Formulates 2 to 4 targeted search queries with domain filters and operators based on the user's objective.
  2. Candidate Retrieval: Collects search results across all formulated queries, deduplicating URLs.
  3. Precision Re-Ranking: Evaluates candidate snippets to reject clickbait, advertisements, and spam, selecting the most authoritative sources.
  4. Deep Fetch and Grounding: Fetches full page content, extracts clean text, and generates verified citations.

Key tools:
- `web_search`: Query the open web for titles, URLs, and snippets.
- `web_fetch`: Fetch a web page and convert its content into clean Markdown.
- `web_research_and_ingest`: Execute multi-query research, fetch top pages, and compile grounded findings with citations.

---

## Cognitive Memory Engine (Loci-Hindsight)

WorkOS embeds a cognitive memory engine backed by SQLite with FTS5 full-text indexing, structured around a spatial Loci architecture:

### Four Memory Networks

1. **Facts**: Verified domain knowledge, reference data, and system facts.
2. **Experiences**: Episodic task logs, past execution outcomes, and procedural learnings.
3. **Entities**: Structured dossiers capturing attributes, roles, and summaries of people, projects, tools, and organizations.
4. **Beliefs**: User preferences, behavioral rules, and operational constraints.

### Active Belief Superseding

When a new belief is stored under an existing key within a given Wing and Hall, `supersede_key` marks preceding records with a `superseded_by` pointer. Queries retrieve only currently active beliefs, allowing user preferences to evolve without manual database pruning.

### Spatial Loci Organization

Memories are organized into canonical Wings and Halls:
- `people`: `contacts`, `roles`, `preferences`, `teams`, `general`
- `projects`: `decisions`, `architecture`, `tasks`, `roadmaps`, `general`
- `workflows`: `preferences`, `runs`, `templates`, `policies`, `general`
- `knowledge`: `python`, `databases`, `tools`, `policies`, `general`

The spatial manager compresses the memory hierarchy into a token-efficient index summary (under 200 tokens) that is injected into planning prompts to provide domain awareness without consuming context.

### Session Drawers (MemPalace Pattern)

Multi-turn conversation sessions are maintained in session drawers. Internal reasoning tokens (such as `<think>` blocks) are stripped prior to storage to maintain compact dialogue histories.

### Asynchronous Memory Reflection

After goal execution completes, the planner invokes an asynchronous reflection pass (`reflect_async`). The reflector inspects the goal, plan steps, tool observations, and synthesized response to extract new facts, record episodic experiences, and update entity profiles.

---

## Document Vault and Provenance

The Document Vault (`data/vault/`) ensures audit-grade provenance for all ingested files:

- **Lossless Storage**: Uploaded documents and email attachments are permanently stored in their original binary format.
- **SHA-256 Deduplication**: Files are hashed on ingestion. Duplicate files return existing vault records, avoiding redundant storage while maintaining metadata history.
- **Parallel Representation**: Stored original files are indexed alongside their Docling-parsed Markdown representations and extracted table catalogs.
- **Side-by-Side Citations**: RAG and research responses include structured citation tokens referencing vault document IDs, file paths, and page offsets.

---

## Infrastructure Automation and Reliability

WorkOS includes an automated infrastructure manager (`workos_engine/infra_manager.py`) and clean process lifecycle controls:

- **Hardware-Aware Ollama Management**: Verifies the Ollama inference endpoint on startup. If offline, spawns a detached background daemon (`start_new_session=True`) configured with hardware-safe flags:
  - `OLLAMA_FLASH_ATTENTION=0`: disables unsupported attention mechanisms on consumer GPUs (such as NVIDIA GTX 1650).
  - `OLLAMA_KV_CACHE_TYPE=f16`: uses native FP16 KV cache to avoid CPU dequantization bottlenecks.
  - `OLLAMA_NUM_PARALLEL=1`: isolates inference memory.
  - Enforces context windows between 6,144 and 12,288 tokens with 500MB VRAM headroom.
- **Docker Service Orchestration**: Verifies Docker daemon connectivity and executes `docker compose up -d` in the background to ensure Elasticsearch (port 9200) and SearXNG (port 8080) are running.
- **Port Conflict Resolution**: Probes port 8000 on launch, detects stale or suspended processes via socket queries and process lookups, and cleanly terminates them before binding.
- **Signal Handling**: Intercepts `SIGINT` (Ctrl+C), `SIGTERM`, and `SIGTSTP` (Ctrl+Z) to release network sockets immediately and prevent lingering zombie processes.

---

## Operational Interfaces

### Interactive Terminal Console (`main.py`)

The Rich-powered interactive CLI provides command parsing, plan visualization, live execution progress, and direct goal execution:

```bash
# Start interactive console
python main.py

# Execute a goal directly in headless mode
python main.py --goal "Search mailbox for invoices from Arvind, parse PDF tables, and summarize totals"

# Preview execution plan without running steps
python main.py --goal "Research recent advancements in small language models" --plan-only

# Override autonomy level for a session
python main.py --goal "Organize inbox" --autonomy FULL
```

#### Slash Commands Reference

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `/help` | | Display slash command reference table |
| `/models` | | List available local Ollama models and active selection |
| `/model` | `<name>` | Switch active generative model at runtime |
| `/policy` | `[FULL \| SUPERVISED]` | View or update system autonomy policy level |
| `/plan` | `<goal>` | Generate and inspect execution DAG without running steps |
| `/session` | `status` | Display active session ID and turn count |
| `/session` | `new` | Initialize a new conversation session drawer |
| `/session` | `clear` | Clear dialogue history in the current session |
| `/session` | `list` | List all stored conversation sessions |
| `/session` | `history` | Print complete dialogue history for active session |
| `/memory` | `summary` | Display Spatial Loci memory index |
| `/memory search` | `<query>` | Search 4-network cognitive memory using FTS5 BM25 |
| `/memory beliefs` | `[wing]` | List active beliefs and behavioral rules |
| `/memory add` | `<wing> <hall> <key> <content>` | Manually store a verified fact into memory |
| `/mail status` | | Check IMAP and SMTP configuration and connectivity |
| `/mail search` | `<query>` | Search mailbox messages via IMAP |
| `/mail draft` | `<to> <subject> <body>` | Stage an email draft in the mailbox |
| `/mail send` | `<to> <subject> <body>` | Send an email (subject to autonomy policy) |
| `/mail organize` | `<category> [sender] [since]` | Organize emails into a category folder |
| `/doc parse` | `<file_path>` | Parse document into Markdown using Docling |
| `/doc tables` | `<file_path>` | Extract structured tables using TableFormer |
| `/rag search` | `<query>` | Execute hybrid RRF search across knowledge base |
| `/rag ingest` | `<file_path>` | Ingest document into parent-child hybrid index |
| `/rag ask` | `<query>` | Query knowledge base with grounded citations |
| `/clear` | | Clear terminal screen |
| `/exit` | | Exit WorkOS console |

### Web Dashboard and REST API (`ui.py`)

WorkOS provides a FastAPI backend and browser-based interface running on port 8000:

```bash
# Start Web UI with automated infrastructure verification
python ui.py

# Or via main CLI
python main.py --ui --port 8000

# Or via launcher script
./start.sh
```

#### Core API Endpoints

- `GET /api/status`: High-level engine health, active model, autonomy level, document count, and agent registry.
- `POST /api/chat`: Conversational endpoint that routes user input to either direct chat or full agent DAG execution.
- `POST /api/plan`: Formulates and returns an execution DAG for a goal without executing steps.
- `POST /api/execute`: Executes a complete goal pipeline, returning step outcomes, artifacts, final synthesis, and citations.
- `GET /api/models`: Lists installed Ollama models with context windows and recommendations.
- `POST /api/model`: Hot-swaps the active model.
- `POST /api/policy`: Updates system autonomy policy (`FULL` or `SUPERVISED`).
- `GET /api/vault/documents`: Lists all permanently stored documents with hash and chunk metadata.
- `POST /api/vault/upload`: Uploads a document to the vault, parses it with Docling, and indexes it into Elasticsearch.
- `GET /api/vault/{doc_id}`: Retrieves document metadata and extracted Markdown.
- `GET /api/vault/{doc_id}/file`: Serves the original raw file for in-browser preview or download.
- `GET /api/memory`: Searches cognitive memory or lists active beliefs and spatial summaries.
- `POST /api/memory`: Stores a fact, belief, experience, or entity into cognitive memory.
- `GET /api/sessions`: Lists active conversation session drawers.
- `GET /api/session/{session_id}`: Retrieves ordered conversation turns for a session.
- `GET /api/infra/status`: Real-time health probe for Ollama, Elasticsearch, SearXNG, and Docker daemon.
- `POST /api/infra/restart`: Triggers auto-start for any offline backing services.
- `GET /api/debug/events`: Retrieves structured execution trace events with filtering by level, type, or session.
- `GET /api/debug/download`: Downloads the full `workos_debug.log` file.

### Centralized Debugging and Tracing (`workos_engine/debug.py`)

WorkOS provides real-time telemetry across all system subsystems:

- **Dual-Destination Logging**: Records events to an in-memory circular ring buffer (1,000 events) for UI inspection while streaming structured JSON entries to `data/logs/workos_debug.log`.
- **Event Types**: Captures `LLM_PROMPT`, `LLM_RESPONSE`, `JSON_PARSE`, `PLAN_FORMULATED`, `STEP_START`, `STEP_COMPLETE`, `TOOL_CALL`, `TOOL_RESULT`, `MEMORY_OP`, `CHAT_INTENT`, `SYSTEM_ERROR`, and `INFRA_EVENT`.
- **Tracing Metadata**: Every event records timestamp, component identifier, log level, session ID, execution duration in milliseconds, structured payload, and stack traces on error.

---

## Configuration Reference

The system is configured via environment variables or a `.env` file in the project root:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for local Ollama inference API |
| `MODEL_NAME` | `refinedneuro/refinedtoolcallv5-3b` | Primary tool-calling LLM model name |
| `EMBEDDING_MODEL` | `bge-m3:latest` | Dense vector embedding model for RAG |
| `DEFAULT_TEMPERATURE` | `0.6` | Generation temperature |
| `TOP_P` | `0.95` | Generation top-p sampling parameter |
| `REPEAT_PENALTY` | `1.1` | Generation repetition penalty |
| `NUM_CTX` | `6144` | Model context window size in tokens |
| `LLM_PROVIDER` | `ollama` | Active LLM provider (`ollama`, `gemini`, or `mock`) |
| `GEMINI_API_KEY` | *(empty)* | API key for optional Google Gemini cloud provider |
| `GEMINI_MODEL` | `gemini-flash-latest` | Model name when Gemini provider is enabled |
| `AUTONOMY_LEVEL` | `SUPERVISED` | System autonomy policy (`SUPERVISED` or `FULL`) |
| `AUTONOMOUS_TRUSTED_RECIPIENTS` | *(empty)* | Comma-separated email addresses exempt from supervision |
| `WORKOS_MEMORY_DB` | `workos_memory.db` | File path for SQLite cognitive memory database |
| `WORKOS_VAULT_DIR` | `data/vault` | Storage directory for original documents and parsed markdown |
| `WORKOS_AUTO_START_INFRA` | `true` | Automatically boot backing infrastructure on launch |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch HTTP endpoint URL |
| `ELASTICSEARCH_USERNAME` | `elastic` | Elasticsearch basic authentication username |
| `ELASTICSEARCH_PASSWORD` | *(empty)* | Elasticsearch basic authentication password |
| `RAG_INDEX_NAME` | `workos_knowledge_base` | Target Elasticsearch index name |
| `SEARXNG_URL` | `http://localhost:8080` | Endpoint for self-hosted SearXNG instance |
| `IMAP_HOST` | *(empty)* | Hostname for incoming IMAP mail server (such as `imap.gmail.com`) |
| `IMAP_PORT` | `993` | Port for incoming IMAP mail server |
| `IMAP_USER` | *(empty)* | Username or email address for IMAP authentication |
| `IMAP_PASSWORD` | *(empty)* | Password or application password for IMAP authentication |
| `SMTP_HOST` | *(empty)* | Hostname for outgoing SMTP mail server (such as `smtp.gmail.com`) |
| `SMTP_PORT` | `587` | Port for outgoing SMTP mail server |
| `SMTP_USER` | *(empty)* | Username or email address for SMTP authentication |
| `SMTP_PASSWORD` | *(empty)* | Password or application password for SMTP authentication |

---

## Quick Start Commands

### 1. Download Required Local Models

```bash
# Pull primary tool-calling model
ollama pull refinedneuro/refinedtoolcallv5-3b

# Pull dense vector embedding model
ollama pull bge-m3:latest
```

### 2. Run the System

```bash
# Start the Web UI dashboard on http://localhost:8000
# (Automatically starts Docker containers and Ollama daemon if offline)
python ui.py

# Or start the interactive terminal console
python main.py

# Or run a goal directly in headless mode
python main.py --goal "Search mailbox for invoices, extract line items with TableFormer, and index into RAG"
```

### 3. Verify System Operations

```bash
# Run end-to-end API simulation suite across all subagents and endpoints
python simulate_ui.py
```
