# MailDoc Multi-Agent

Autonomous email triage, PDF decryption, and hybrid RAG assistant built with AutoGen and connected via Model Context Protocol (MCP).

## Connected Repositories

This orchestrator connects to two standalone MCP services:
* **MailDoc MCP Server**: [github.com/Mohking1/maildoc-mcp](https://github.com/Mohking1/maildoc-mcp) — IMAP email operations and PDF text extraction.
* **Hybrid RAG MCP Server**: [github.com/Mohking1/Hybrid-RAG](https://github.com/Mohking1/Hybrid-RAG) — Elasticsearch 8 Parent-Child hybrid retrieval engine.

## What it does

1. Connects to the **MailDoc MCP Server** (`../maildoc/server.py`) to search, read, and manage Gmail via IMAP.
2. Identifies password-protected PDF attachments and inspects email body text for password hints (DOB, PAN, dates, passcode patterns).
3. Connects to the **Hybrid RAG MCP Server** (`../RAG/mcp_server.py`) to ingest PDFs into an Elasticsearch 8 knowledge base (Parent-Child chunking + BM25 + dense vectors + cross-encoder reranker).
4. Answers user queries with page-level document citations.
5. Saves long-term user preferences with Mem0.

## Setup

Clone the sibling repositories into the same parent directory:

```bash
git clone https://github.com/Mohking1/maildoc-mcp.git ../maildoc
git clone https://github.com/Mohking1/Hybrid-RAG.git ../RAG

pip install -r requirements.txt
pip install -r ../maildoc/requirements.txt
pip install -r ../RAG/requirements.txt
```

Set credentials in `.env`:

```env
GEMINI_API_KEY=
GMAIL_USERNAME=
GMAIL_APP_PASSWORD=
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=
```

Start Elasticsearch (if using local Docker):

```bash
cd ../RAG && docker compose up -d && cd ../Multi-Agent
```

## Usage

```bash
python main_agent.py
```
