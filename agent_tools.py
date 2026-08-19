import json
import logging
import os
import re
import tempfile
from typing import Any

try:
    from mem0 import Memory
except ImportError:
    Memory = None

from mcp_client import mcp_manager

logger = logging.getLogger("multi_agent.tools")

# Memory initialization
try:
    memory = Memory() if Memory is not None else None
except Exception as e:  # noqa: BLE001
    logger.warning("Memory initialization failed: %s", e)
    memory = None


# --- MailDoc Tools (Dispatched via MailDoc MCP Server) ---


def search_emails(
    sender: str | None = None,
    subject: str | None = None,
    body: str | None = None,
    seen: bool | None = None,
    date_gte: str | None = None,
    date_lt: str | None = None,
    with_attachments: bool = False,
    folder: str = "INBOX",
    limit: int = 10,
) -> Any:
    """Search emails using the MailDoc MCP server."""
    args = {
        "sender": sender,
        "subject": subject,
        "body": body,
        "seen": seen,
        "date_gte": date_gte,
        "date_lt": date_lt,
        "with_attachments": with_attachments,
        "folder": folder,
        "limit": limit,
    }
    return mcp_manager.call_tool_sync("maildoc", "search_emails", args)


def fetch_email_by_uid(uid: str, folder: str = "INBOX") -> Any:
    """Fetch email details by UID via MailDoc MCP server."""
    return mcp_manager.call_tool_sync(
        "maildoc", "fetch_email_by_uid", {"uid": uid, "folder": folder}
    )


def download_attachment(
    uid: str, attachment_filename: str, folder: str = "INBOX"
) -> Any:
    """Download attachment from email via MailDoc MCP server."""
    return mcp_manager.call_tool_sync(
        "maildoc",
        "download_attachment",
        {"uid": uid, "attachment_filename": attachment_filename, "folder": folder},
    )


def extract_pdf_text(pdf_bytes: bytes, password: str | None = None) -> str:
    """Extract full text from PDF bytes via MailDoc MCP server."""
    return str(
        mcp_manager.call_tool_sync(
            "maildoc",
            "extract_pdf_text",
            {"pdf_bytes": pdf_bytes, "password": password},
        )
    )


def extract_pdf_page_text(
    pdf_bytes: bytes, page_number: int, password: str | None = None
) -> str:
    """Extract page text from PDF via MailDoc MCP server."""
    return str(
        mcp_manager.call_tool_sync(
            "maildoc",
            "extract_pdf_page_text",
            {"pdf_bytes": pdf_bytes, "page_number": page_number, "password": password},
        )
    )


def read_pdf_attachment(
    uid: str,
    attachment_filename: str,
    password: str | None = None,
    folder: str = "INBOX",
) -> str:
    """Read text from an email PDF attachment via MailDoc MCP server."""
    return str(
        mcp_manager.call_tool_sync(
            "maildoc",
            "read_pdf_attachment",
            {
                "uid": uid,
                "attachment_filename": attachment_filename,
                "password": password,
                "folder": folder,
            },
        )
    )


def mark_email_as_read(uid: str, folder: str = "INBOX") -> Any:
    """Mark an email as read via MailDoc MCP server."""
    return mcp_manager.call_tool_sync(
        "maildoc", "mark_email_as_read", {"uid": uid, "folder": folder}
    )


def move_email(uid: str, destination_folder: str, source_folder: str = "INBOX") -> Any:
    """Move email to target folder via MailDoc MCP server."""
    return mcp_manager.call_tool_sync(
        "maildoc",
        "move_email",
        {
            "uid": uid,
            "destination_folder": destination_folder,
            "source_folder": source_folder,
        },
    )


def delete_email(uid: str, folder: str = "INBOX") -> Any:
    """Delete an email via MailDoc MCP server."""
    return mcp_manager.call_tool_sync(
        "maildoc", "delete_email", {"uid": uid, "folder": folder}
    )


def create_folder(folder_name: str) -> Any:
    """Create mailbox folder via MailDoc MCP server."""
    return mcp_manager.call_tool_sync(
        "maildoc", "create_folder", {"folder_name": folder_name}
    )


def delete_folder(folder_name: str) -> Any:
    """Delete mailbox folder via MailDoc MCP server."""
    return mcp_manager.call_tool_sync(
        "maildoc", "delete_folder", {"folder_name": folder_name}
    )


def list_folders() -> Any:
    """List mailbox folders via MailDoc MCP server."""
    return mcp_manager.call_tool_sync("maildoc", "list_folders")


# --- PDF Security & Heuristic Tools ---


def check_pdf_encryption(
    uid: str, attachment_filename: str, folder: str = "INBOX"
) -> dict[str, Any]:
    """Check if an email attachment is an encrypted PDF."""
    raw = download_attachment(uid, attachment_filename, folder=folder)
    if not raw:
        return {
            "status": "error",
            "error": f"Attachment '{attachment_filename}' not found.",
        }

    # Handle bytes or string payload
    pdf_bytes = raw.encode() if isinstance(raw, str) else raw
    try:
        import pymupdf

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            return {
                "status": "success",
                "filename": attachment_filename,
                "is_encrypted": doc.is_encrypted,
                "pages": len(doc) if not doc.is_encrypted else 0,
            }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


def extract_password_hints_from_email(
    uid: str, folder: str = "INBOX"
) -> dict[str, Any]:
    """Inspect email text for potential attachment passwords."""
    raw_email = fetch_email_by_uid(uid, folder=folder)
    if not raw_email:
        return {"status": "error", "error": f"Could not fetch email {uid}."}

    email_data = raw_email if isinstance(raw_email, dict) else {}
    if isinstance(raw_email, str):
        try:
            email_data = json.loads(raw_email)
        except json.JSONDecodeError:
            email_data = {"text": raw_email}

    content = f"{email_data.get('subject', '')} {email_data.get('text', '')} {email_data.get('html', '')}"
    hints = []

    # Explicit passcode patterns
    explicit = re.findall(
        r"(?:password|passcode|pin|pwd)\s*(?:is|:|=|-)?\s*([A-Za-z0-9@#$%^&*!_-]{4,20})",
        content,
        re.IGNORECASE,
    )
    for match in explicit:
        cleaned = match.strip().strip(".,;:\"'")
        if cleaned and cleaned not in hints:
            hints.append(cleaned)

    # Date patterns (DDMMYYYY, YYYYMMDD, DD-MM-YYYY)
    dates = re.findall(
        r"\b(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2}|\d{8})\b", content
    )
    for d in dates:
        cleaned = re.sub(r"[/-]", "", d)
        if cleaned not in hints:
            hints.append(cleaned)

    # Standard alphanumeric IDs (e.g. PAN)
    pan_matches = re.findall(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", content)
    for p in pan_matches:
        if p not in hints:
            hints.append(p)

    return {
        "status": "success",
        "uid": uid,
        "hints": hints,
    }


def unlock_pdf_attachment(
    uid: str, attachment_filename: str, password: str, folder: str = "INBOX"
) -> dict[str, Any]:
    """Test candidate password against an encrypted PDF attachment."""
    raw = download_attachment(uid, attachment_filename, folder=folder)
    if not raw:
        return {
            "status": "error",
            "error": f"Attachment '{attachment_filename}' not found.",
        }

    pdf_bytes = raw.encode() if isinstance(raw, str) else raw
    try:
        import pymupdf

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if not doc.is_encrypted:
                return {
                    "status": "success",
                    "unlocked": True,
                    "message": "PDF is not encrypted.",
                }

            if doc.authenticate(password):
                return {
                    "status": "success",
                    "unlocked": True,
                    "pages": len(doc),
                }
            return {
                "status": "failed",
                "unlocked": False,
                "error": "Incorrect password.",
            }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


# --- RAG Tools (Dispatched via RAG MCP Server) ---


def search_knowledge_base(
    query: str, collection_name: str = "default", top_k: int = 5
) -> str:
    """Ask a question and receive a citation-grounded answer via RAG MCP server."""
    result = mcp_manager.call_tool_sync(
        "rag",
        "rag_ask",
        {"question": query, "collection_name": collection_name, "top_k": top_k},
    )
    return str(result)


def rag_search(query: str, collection_name: str = "default", top_k: int = 5) -> str:
    """Retrieve relevant parent context chunks and scores via RAG MCP server."""
    result = mcp_manager.call_tool_sync(
        "rag",
        "rag_search",
        {"query": query, "collection_name": collection_name, "top_k": top_k},
    )
    return str(result)


def rag_route_query(query: str, collection_name: str = "default") -> str:
    """Inspect query complexity metrics and routing tier via RAG MCP server."""
    result = mcp_manager.call_tool_sync(
        "rag", "rag_route_query", {"query": query, "collection_name": collection_name}
    )
    return str(result)


def process_email_attachment_to_rag(
    email_uid: str,
    attachment_filename: str,
    password: str | None = None,
    collection_name: str = "default",
) -> str:
    """Download attachment, decrypt if necessary, and ingest via RAG MCP server."""
    raw = download_attachment(email_uid, attachment_filename)
    if not raw:
        return f"Failed to download attachment '{attachment_filename}'."

    pdf_bytes = raw.encode() if isinstance(raw, str) else raw
    temp_path = None
    try:
        import pymupdf

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                if not password:
                    return (
                        f"PDF '{attachment_filename}' is encrypted. "
                        "Check email text with extract_password_hints_from_email or provide password."
                    )
                if not doc.authenticate(password):
                    return (
                        f"Failed to unlock '{attachment_filename}': Incorrect password."
                    )

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    temp_path = tmp.name
                doc.save(temp_path, encryption=pymupdf.PDF_ENCRYPT_NONE)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    temp_path = tmp.name

        # Ingest into RAG via MCP tool
        result = mcp_manager.call_tool_sync(
            "rag",
            "rag_ingest_pdf",
            {
                "pdf_path": temp_path,
                "collection_name": collection_name,
                "doc_id": attachment_filename,
            },
        )
        return f"Ingestion result for '{attachment_filename}':\n{result}"

    except Exception as e:  # noqa: BLE001
        return f"Processing error: {e}"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# --- Long-Term Memory Tools ---


def add_user_memory(user_id: str, text: str) -> str:
    """Store user preference or contextual detail in long-term memory."""
    if not memory:
        return "Memory storage unavailable."
    try:
        memory.add(text, user_id=user_id)
        return f"Stored memory for user '{user_id}'."
    except Exception as e:  # noqa: BLE001
        return f"Failed to store memory: {e}"


def get_user_memories(user_id: str, query: str) -> list[str]:
    """Retrieve relevant memories for a user query."""
    if not memory:
        return ["Memory storage unavailable."]
    try:
        results = memory.search(query, user_id=user_id)
        if isinstance(results, str):
            return [results]
        if isinstance(results, list):
            memories = []
            for item in results:
                if isinstance(item, dict) or hasattr(item, "get"):
                    memories.append(item.get("memory", item.get("text", str(item))))
                else:
                    memories.append(str(item))
            return memories if memories else ["No matching memories found."]
        return [str(results)]
    except Exception as e:  # noqa: BLE001
        return [f"Failed to fetch memories: {e}"]


ALL_TOOLS = [
    # MailDoc MCP tools
    search_emails,
    fetch_email_by_uid,
    download_attachment,
    extract_pdf_text,
    extract_pdf_page_text,
    read_pdf_attachment,
    mark_email_as_read,
    move_email,
    delete_email,
    create_folder,
    delete_folder,
    list_folders,
    # PDF security & heuristics
    check_pdf_encryption,
    extract_password_hints_from_email,
    unlock_pdf_attachment,
    # RAG MCP tools
    search_knowledge_base,
    rag_search,
    rag_route_query,
    process_email_attachment_to_rag,
    # Memory
    add_user_memory,
    get_user_memories,
]
