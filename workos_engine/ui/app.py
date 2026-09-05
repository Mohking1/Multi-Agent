"""FastAPI Backend Server for Argus OS Executive AI Web Interface."""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import WorkOSConfig, get_config
from workos_engine import __version__
from workos_engine.debug import get_tracer
from workos_engine.infra_manager import (
    ensure_all_infrastructure,
    get_infrastructure_status,
)
from workos_engine.memory.networks import CognitiveMemoryEngine
from workos_engine.planner import ExecutivePlanner
from workos_engine.types import AutonomyLevel, MemoryNetwork
from workos_engine.vault import DocumentVault

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure backing infrastructure is running when app starts
    if getattr(config, "auto_start_infrastructure", True):
        try:
            ensure_all_infrastructure(verbose=False)
        except Exception as e:
            logger.warning(f"Startup infrastructure initialization error: {e}")
    yield


app = FastAPI(
    title="Argus OS — Executive AI Operating System",
    description="Executive Autonomous Multi-Agent AI Workspace with Live Telemetry & Mission Control",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global WorkOS engine instances
config: WorkOSConfig = get_config()
vault: DocumentVault | None = None


def get_vault() -> DocumentVault:
    global vault
    if vault is not None:
        return vault
    target_dir = os.getenv("WORKOS_VAULT_DIR") or config.vault_dir
    return DocumentVault(vault_dir=target_dir)


planner: ExecutivePlanner = ExecutivePlanner(config=config)
memory: CognitiveMemoryEngine = planner.memory

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Pydantic Schemas
class GoalRequest(BaseModel):
    goal: str
    plan_only: bool = False
    session_id: str | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class PolicyUpdateRequest(BaseModel):
    autonomy_level: str


class ModelUpdateRequest(BaseModel):
    model_name: str


class MemoryAddRequest(BaseModel):
    network: str
    wing: str
    hall: str
    key: str
    content: str
    confidence: float = 1.0


class EmailSendRequest(BaseModel):
    to_email: str
    subject: str
    body: str
    cc: str | None = None
    bcc: str | None = None


class SessionTitleRequest(BaseModel):
    title: str


class DraftApproveRequest(BaseModel):
    uid: str | None = None
    to_email: str
    subject: str
    body: str


# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Argus OS</h1><p>static/index.html not found.</p>")


@app.get("/api/stats")
async def get_workspace_stats():
    """Aggregated executive workspace statistics and system intelligence."""
    docs = get_vault().list_documents()
    total_size = sum(d.file_size_bytes for d in docs)
    parsed_count = sum(1 for d in docs if d.parsed_markdown_path and Path(d.parsed_markdown_path).exists())
    total_tables = sum(d.table_count for d in docs)

    active_beliefs = []
    wings_count = 0
    halls_count = 0
    try:
        active_beliefs = memory.get_active_beliefs()
        loci_data = memory.db.get_loci_structure(active_only=True)
        wings_count = len(loci_data)
        halls_count = sum(len(h) for h in loci_data.values())
    except Exception as e:
        logger.debug(f"Error fetching memory stats: {e}")

    ollama_client = getattr(planner, "client", None)
    is_ollama_online = False
    if ollama_client and hasattr(ollama_client, "is_available"):
        is_ollama_online = ollama_client.is_available()

    size_formatted = f"{(total_size / (1024 * 1024)):.2f} MB" if total_size > 1024 * 1024 else f"{(total_size / 1024):.1f} KB"

    return {
        "vault": {
            "document_count": len(docs),
            "parsed_count": parsed_count,
            "table_count": total_tables,
            "total_size_bytes": total_size,
            "total_size_formatted": size_formatted,
        },
        "memory": {
            "active_beliefs_count": len(active_beliefs),
            "wings_count": wings_count,
            "halls_count": halls_count,
        },
        "mail": {
            "connected": bool(getattr(config, "imap_server", None)),
            "autonomy_level": config.autonomy_level.value,
        },
        "system": {
            "system_name": "Argus OS",
            "version": __version__,
            "model_name": config.model_name,
            "num_ctx": config.num_ctx,
            "ollama_online": is_ollama_online,
        },
    }


@app.get("/api/stream/{session_id}")
async def stream_subagent_events(session_id: str):
    """
    Real-time Server-Sent Events (SSE) stream yielding subagent thoughts,
    tool calls, and execution milestones directly to the live Mission Control UI.
    """
    tracer = get_tracer()
    queue = tracer.subscribe(session_id=session_id)

    async def event_generator():
        yield f"event: connected\ndata: {json.dumps({'status': 'connected', 'session_id': session_id})}\n\n"
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    event_data = event.to_dict()
                    yield f"event: trace\ndata: {json.dumps(event_data)}\n\n"
                except TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            tracer.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.put("/api/session/{session_id}/title")
async def update_session_title(session_id: str, req: SessionTitleRequest):
    """Updates the human-readable title of a session."""
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    clean_title = req.title.strip()
    memory.set_session_title(session_id=session_id, title=clean_title)
    return {"status": "success", "session_id": session_id, "title": clean_title}


@app.get("/api/status")
async def get_system_status():
    """Returns high-level WorkOS engine health, models, and vault metrics."""
    ollama_client = getattr(planner, "client", None)
    is_ollama_online = False
    available_models = []

    if ollama_client and hasattr(ollama_client, "is_available"):
        is_ollama_online = ollama_client.is_available()
        if is_ollama_online:
            available_models = ollama_client.list_models()

    docs = get_vault().list_documents()
    active_beliefs = memory.get_active_beliefs()

    return {
        "version": __version__,
        "model_name": config.model_name,
        "embedding_model": config.embedding_model,
        "ollama_base_url": config.ollama_base_url,
        "ollama_online": is_ollama_online,
        "available_models": available_models,
        "autonomy_level": config.autonomy_level.value,
        "trusted_recipients": config.trusted_recipients,
        "documents_count": len(docs),
        "active_beliefs_count": len(active_beliefs),
        "num_ctx": config.num_ctx,
        "agents": list(planner.agents.keys()),
    }


@app.post("/api/policy")
async def update_policy(req: PolicyUpdateRequest):
    level_str = req.autonomy_level.upper().strip()
    if level_str in ("AUTO", "AUTONOMOUS", "FULL"):
        new_level = AutonomyLevel.FULL
    elif level_str in ("SUPERVISED", "MANUAL", "GUARDED"):
        new_level = AutonomyLevel.SUPERVISED
    else:
        raise HTTPException(status_code=400, detail=f"Invalid autonomy level: {req.autonomy_level}")

    config.autonomy_level = new_level
    planner.config.autonomy_level = new_level
    planner.mail_agent.config.autonomy_level = new_level
    return {"status": "success", "autonomy_level": new_level.value}


@app.get("/api/models")
async def get_available_models():
    """Returns the list of installed and supported Ollama models with metadata and recommendations."""
    models = []
    try:
        import httpx

        with httpx.Client(timeout=5.0) as client:
            res = client.get(f"{config.ollama_base_url}/api/tags")
            if res.status_code == 200:
                data = res.json()
                for m in data.get("models", []):
                    name = m.get("name")
                    if name:
                        models.append(name)
    except Exception as e:
        logger.debug(f"Failed to fetch Ollama tags: {e}")

    if not models:
        models = [
            "refinedneuro/refinedtoolcallv5-3b",
            "hf.co/unsloth/SmolLM3-3B-GGUF:UD-Q6_K_XL",
            "hf.co/Qwen/Qwen3-1.7B-GGUF:Q8_0",
        ]

    structured = []
    for m in models:
        is_primary = "refinedtoolcall" in m.lower()
        is_smol = "smollm" in m.lower()
        structured.append(
            {
                "id": m,
                "name": "RefinedToolCallV5 (3.1B)"
                if is_primary
                else ("SmolLM3 (3.1B)" if is_smol else m.split("/")[-1].split(":")[0]),
                "recommended": is_primary,
                "tag": "★ Recommended (Precision Tool Calling & DAG)"
                if is_primary
                else ("Conversational Specialist" if is_smol else "Lightweight / Fast"),
                "context": f"{config.num_ctx:,} tokens (Safe 12K Window - 500MB VRAM Headroom)",
                "active": m == config.model_name
                or (is_primary and "refinedtoolcall" in config.model_name.lower()),
            }
        )

    return {
        "models": models,
        "active_model": config.model_name,
        "num_ctx": config.num_ctx,
        "detailed_models": structured,
    }


@app.post("/api/model")
async def update_model(req: ModelUpdateRequest):
    if not req.model_name.strip():
        raise HTTPException(status_code=400, detail="Model name cannot be empty")

    config.model_name = req.model_name.strip()
    planner.config.model_name = config.model_name
    if hasattr(planner.client, "default_model"):
        planner.client.default_model = config.model_name
    return {"status": "success", "model_name": config.model_name}


@app.get("/api/infra/status")
async def get_infra_status_endpoint():
    """Returns real-time health of Ollama, Elasticsearch, SearXNG, and Docker."""
    return get_infrastructure_status()


@app.post("/api/infra/restart")
async def restart_infra_endpoint():
    """Triggers auto-start/restart for any offline infrastructure services."""
    status = ensure_all_infrastructure(verbose=False)
    return status


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Conversational chat endpoint — routes between conversational and agent execution."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    sess_id = req.session_id or "web_session"
    try:
        result = await planner.chat_turn(message=req.message, session_id=sess_id)
        return result
    except Exception as e:
        logger.exception(f"Chat turn error: {e}")
        from workos_engine.debug import get_tracer

        get_tracer().log_error(
            "ui.chat_endpoint", e, context={"message": req.message, "session_id": sess_id}
        )
        return {
            "type": "error",
            "message": f"Execution notice: {e}",
            "session_id": sess_id,
            "error": str(e),
        }


@app.get("/api/debug/events")
async def get_debug_events_endpoint(
    limit: int = 100,
    event_type: str | None = None,
    session_id: str | None = None,
    level: str | None = None,
    search: str | None = None,
):
    """Fetches real-time structured debug trace events."""
    from workos_engine.debug import get_tracer

    tracer = get_tracer()
    return {
        "summary": tracer.get_summary(),
        "events": tracer.get_events(
            limit=limit, event_type=event_type, session_id=session_id, level=level, search=search
        ),
    }


@app.post("/api/debug/clear")
async def clear_debug_events_endpoint():
    """Clears in-memory debug trace buffer."""
    from workos_engine.debug import get_tracer

    cleared = get_tracer().clear()
    return {"status": "cleared", "count": cleared}


@app.get("/api/debug/download")
async def download_debug_log_endpoint():
    """Downloads the full workos_debug.log file."""
    from workos_engine.debug import get_tracer

    log_path = get_tracer().log_file
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Debug log file not yet created.")
    return FileResponse(
        path=str(log_path),
        filename="workos_debug.log",
        media_type="text/plain",
    )


@app.post("/api/plan")
async def generate_plan_endpoint(req: GoalRequest):
    """Generates dynamic execution DAG for a goal without executing it."""
    context = planner.build_planning_context(req.goal, session_id=req.session_id)
    plan = planner.generate_plan(req.goal, context=context)
    return {
        "goal": plan.goal,
        "steps": [
            {
                "step_id": s.step_id,
                "description": s.description,
                "assigned_agent": s.assigned_agent,
                "input_data": s.input_data,
                "status": s.status,
            }
            for s in plan.steps
        ],
    }


@app.post("/api/execute")
async def execute_goal_endpoint(req: GoalRequest):
    """Full execution pipeline: generates plan, executes subagents, synthesizes grounded answer."""
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")

    sess_id = req.session_id or "web_session"
    if req.plan_only:
        context = planner.build_planning_context(req.goal, session_id=sess_id)
        plan = planner.generate_plan(req.goal, context=context)
        return {"plan": plan, "executed": False}

    plan = await planner.run_goal(req.goal, session_id=sess_id)

    # Collect any citations produced during RAG and Web research steps
    citations = []
    for step in plan.steps:
        if step.result and isinstance(step.result.data, dict):
            if "citations" in step.result.data and isinstance(step.result.data["citations"], list):
                for cit in step.result.data["citations"]:
                    if isinstance(cit, dict) and not any(
                        c.get("citation_id") == cit.get("citation_id") for c in citations
                    ):
                        citations.append(cit)
        elif step.result and isinstance(step.result.data, list):
            # Web search results list
            for idx, r in enumerate(step.result.data, 1):
                if isinstance(r, dict) and (r.get("url") or r.get("title")):
                    cit_id = f"[{idx}]"
                    if not any(c.get("citation_id") == cit_id for c in citations):
                        citations.append(
                            {
                                "citation_id": cit_id,
                                "title": r.get("title", f"Result {idx}"),
                                "filename": r.get("title", f"Result {idx}"),
                                "url": r.get("url", ""),
                                "snippet": r.get("snippet", ""),
                                "score": r.get("score", 1.0),
                            }
                        )
        if step.result and isinstance(step.result.artifacts, list):
            for art in step.result.artifacts:
                if isinstance(art, dict) and "citation_id" in art:
                    if not any(c.get("citation_id") == art.get("citation_id") for c in citations):
                        citations.append(art)

    # Also parse sources lines from final synthesis (e.g. Sources:\n[1] https://...)
    if plan.final_output:
        import re

        for m in re.finditer(r"\[(\d+)\]\s*([^\n\r]+)", plan.final_output):
            cit_id = f"[{m.group(1)}]"
            if not any(c.get("citation_id") == cit_id for c in citations):
                line_content = m.group(2).strip()
                url_match = re.search(r"https?://[^\s\)]+", line_content)
                url = url_match.group(0) if url_match else ""
                title = line_content.replace(url, "").strip(" -–—:") if url else line_content
                citations.append(
                    {
                        "citation_id": cit_id,
                        "title": title or line_content,
                        "filename": title or line_content,
                        "url": url,
                        "snippet": line_content,
                        "score": 1.0,
                    }
                )

    return {
        "goal": plan.goal,
        "final_output": plan.final_output,
        "status": "completed"
        if all(s.status == "completed" for s in plan.steps)
        else "completed_with_issues",
        "citations": citations,
        "steps": [
            {
                "step_id": s.step_id,
                "description": s.description,
                "assigned_agent": s.assigned_agent,
                "input_data": s.input_data,
                "status": s.status,
                "result": {
                    "success": s.result.success if s.result else False,
                    "data": s.result.data if s.result else None,
                    "error": s.result.error if s.result else None,
                    "artifacts": s.result.artifacts if s.result else [],
                }
                if s.result
                else None,
            }
            for s in plan.steps
        ],
    }


@app.get("/api/sessions")
async def list_sessions_endpoint():
    """Lists all active conversation sessions and turn counts."""
    return {"sessions": memory.list_sessions()}


@app.get("/api/session/{session_id}")
async def get_session_endpoint(session_id: str):
    """Retrieves verbatim conversation turns for a session (MemPalace drawer)."""
    turns = memory.get_session_dialogue(session_id=session_id, limit=50)
    return {
        "session_id": session_id,
        "turns": [
            {
                "id": t.id,
                "role": t.role,
                "content": t.content,
                "created_at": t.created_at,
                "metadata": t.metadata,
            }
            for t in turns
        ],
    }


@app.delete("/api/session/{session_id}")
async def clear_session_endpoint(session_id: str):
    """Clears conversation turns for a session."""
    deleted = memory.clear_session(session_id=session_id)
    return {"status": "success", "session_id": session_id, "deleted_turns": deleted}


@app.get("/api/memory")
async def get_memory_data(
    query: str | None = None, wing: str | None = None, network: str | None = None
):
    """Searches or lists cognitive memories and spatial index."""
    if query:
        net = MemoryNetwork(network) if network else None
        results = memory.recall(query=query, wing=wing, network=net, limit=25)
        return {
            "query": query,
            "results": [
                {
                    "id": m.id,
                    "network": m.network.value if hasattr(m.network, "value") else str(m.network),
                    "wing": m.wing,
                    "hall": m.hall,
                    "key": m.key,
                    "content": m.content,
                    "confidence": m.confidence,
                    "created_at": m.created_at,
                }
                for m in results
            ],
        }

    active_beliefs = memory.get_active_beliefs(wing=wing)
    summary = memory.get_context_index_summary()

    return {
        "spatial_summary": summary,
        "active_beliefs": [
            {
                "id": b.id,
                "wing": b.wing,
                "hall": b.hall,
                "key": b.key,
                "content": b.content,
                "created_at": b.created_at,
            }
            for b in active_beliefs
        ],
    }


@app.post("/api/memory")
async def add_memory_item(req: MemoryAddRequest):
    """Adds a new fact, entity, experience, or belief to CognitiveMemory."""
    net = req.network.lower()
    if net == "facts":
        mem_id = memory.retain_fact(
            wing=req.wing,
            hall=req.hall,
            key=req.key,
            content=req.content,
            confidence=req.confidence,
        )
    elif net == "beliefs":
        mem_id = memory.retain_belief(
            wing=req.wing, hall=req.hall, key=req.key, content=req.content
        )
    elif net == "experiences":
        mem_id = memory.retain_experience(
            wing=req.wing, hall=req.hall, key=req.key, content=req.content
        )
    else:
        mem_id = memory.retain_fact(wing=req.wing, hall=req.hall, key=req.key, content=req.content)

    return {"status": "success", "id": mem_id}


@app.get("/api/vault")
@app.get("/api/vault/documents")
async def list_vault_documents():
    """Lists all permanently stored original documents in the Vault."""
    docs = get_vault().list_documents()
    return {
        "documents": [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "file_size_bytes": d.file_size_bytes,
                "mime_type": d.mime_type,
                "sha256_hash": d.sha256_hash,
                "created_at": d.created_at,
                "table_count": d.table_count,
                "chunk_count": d.chunk_count,
                "has_parsed": bool(
                    d.parsed_markdown_path and Path(d.parsed_markdown_path).exists()
                ),
            }
            for d in docs
        ]
    }


@app.get("/api/vault/{doc_id}")
async def get_vault_document_details(doc_id: str):
    """Retrieves document metadata, original file info, and parsed markdown content."""
    doc = get_vault().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in vault")

    parsed_md = get_vault().get_parsed_content(doc_id)
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "file_size_bytes": doc.file_size_bytes,
        "mime_type": doc.mime_type,
        "sha256_hash": doc.sha256_hash,
        "vault_path": doc.vault_path,
        "created_at": doc.created_at,
        "metadata": doc.metadata,
        "parsed_markdown": parsed_md,
    }


@app.get("/api/vault/{doc_id}/file")
async def download_vault_file(doc_id: str):
    """Serves the original raw stored file for in-browser preview or download."""
    doc = get_vault().get_document(doc_id)
    if not doc or not Path(doc.vault_path).exists():
        raise HTTPException(status_code=404, detail="Original document file not found")

    return FileResponse(
        path=doc.vault_path,
        filename=doc.filename,
        media_type=doc.mime_type,
    )


@app.post("/api/vault/upload")
async def upload_document_to_vault(
    file: UploadFile = File(...), parse_and_index: bool = Form(True)
):
    """Uploads an original document to the Vault, triggers Docling parsing, and indexes into RAG."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 1. Store in vault with preserved original filename
        vault_doc = get_vault().store_document(
            tmp_path,
            filename=file.filename,
            metadata={"original_upload_name": file.filename},
        )

        # 2. Parse with Docling & Index into RAG if requested
        if parse_and_index:
            try:
                planner.doc_agent.toolkit.parse_document(
                    vault_doc.vault_path, doc_id=vault_doc.doc_id
                )
                planner.rag_agent.toolkit.rag_ingest_pdf(
                    vault_doc.vault_path, doc_id=vault_doc.doc_id
                )
            except Exception as e:
                logger.warning(f"Error parsing/indexing uploaded doc: {e}")

        return {
            "status": "success",
            "doc_id": vault_doc.doc_id,
            "filename": vault_doc.filename,
            "file_size": vault_doc.file_size_bytes,
            "vault_path": vault_doc.vault_path,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/mail/inbox")
async def get_mail_inbox(query: str | None = None):
    """Searches or fetches emails from the Mail Specialist agent."""
    try:
        res = planner.mail_agent.toolkit.search_emails(text=query or "")
        emails = res if isinstance(res, list) else res.get("emails", [])
        return {"emails": emails, "count": len(emails)}
    except Exception as e:
        logger.warning(f"Failed to fetch mail inbox: {e}")
        return {"emails": [], "count": 0, "error": str(e)}


@app.post("/api/mail/send")
async def send_mail_endpoint(req: EmailSendRequest):
    """Dispatches or stages an outbound email through the MailAgent policy gate."""
    try:
        res = planner.mail_agent.execute_send(
            to_email=req.to_email,
            subject=req.subject,
            body=req.body,
            cc=req.cc,
            bcc=req.bcc,
        )
        return res
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mail/drafts")
async def list_staged_drafts():
    """Lists staged drafts in IMAP Drafts folder awaiting supervised approval."""
    try:
        drafts = []
        if hasattr(planner.mail_agent, "toolkit") and hasattr(planner.mail_agent.toolkit, "search_emails"):
            res = planner.mail_agent.toolkit.search_emails(folder="Drafts", text="")
            drafts = res if isinstance(res, list) else res.get("emails", [])
        return {"drafts": drafts, "count": len(drafts)}
    except Exception as e:
        logger.debug(f"Failed to fetch drafts: {e}")
        return {"drafts": [], "count": 0}


@app.post("/api/mail/drafts/approve")
async def approve_staged_draft(req: DraftApproveRequest):
    """Approves and transmits a staged email draft via SMTP."""
    try:
        if not hasattr(planner.mail_agent, "toolkit") or not hasattr(planner.mail_agent.toolkit, "send_email"):
            raise HTTPException(status_code=500, detail="Mail agent toolkit not configured")
        res = planner.mail_agent.toolkit.send_email(
            to_email=req.to_email,
            subject=req.subject,
            body=req.body,
        )
        return {"status": "sent", "to_email": req.to_email, "result": res}
    except Exception as e:
        logger.error(f"Failed to transmit approved draft: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/web/search")
async def web_search_endpoint(query: str, num_results: int = 5):
    """Zero-cloud live web search endpoint."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = planner.web_agent.toolkit.search_web(query=query, num_results=num_results)
    return {"query": query, "results": results, "count": len(results)}


def create_ui_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
