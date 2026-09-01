"""Document Vault & Citation Manager for WorkOS.

Handles persistent storage of original uploaded/ingested documents (PDF, DOCX, TXT, attachments),
content deduplication via SHA-256, metadata tracking, and side-by-side citation linking.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VaultDocument:
    doc_id: str
    filename: str
    original_path: str
    vault_path: str
    file_size_bytes: int
    sha256_hash: str
    mime_type: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    table_count: int = 0
    parsed_markdown_path: str | None = None


class DocumentVault:
    """
    Manages persistent storage, cataloging, and retrieval of original source documents
    to guarantee lossless provenance and audit-grade citation grounding.
    """

    def __init__(self, vault_dir: str | None = None):
        self.vault_dir = Path(vault_dir or os.getenv("WORKOS_VAULT_DIR", "data/vault")).resolve()
        self.docs_dir = self.vault_dir / "documents"
        self.parsed_dir = self.vault_dir / "parsed"
        self.catalog_file = self.vault_dir / "catalog.json"

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

        self._catalog: dict[str, VaultDocument] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Loads document catalog from disk."""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for doc_id, item in data.items():
                        self._catalog[doc_id] = VaultDocument(**item)
            except Exception as e:
                logger.warning(f"Failed to load vault catalog: {e}")
                self._catalog = {}

    def _save_catalog(self) -> None:
        """Persists document catalog to disk."""
        try:
            with open(self.catalog_file, "w", encoding="utf-8") as f:
                data = {doc_id: asdict(doc) for doc_id, doc in self._catalog.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save vault catalog: {e}")

    @staticmethod
    def calculate_sha256(file_path: str | Path) -> str:
        """Computes SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def store_document(
        self,
        file_path: str | Path,
        doc_id: str | None = None,
        filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VaultDocument:
        """
        Stores an original document into the vault.
        If the file already exists (by SHA-256 hash), returns the existing entry.
        """
        src_path = Path(file_path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Source document not found: {src_path}")

        file_hash = self.calculate_sha256(src_path)
        final_filename = filename or src_path.name

        # Deduplication check
        for doc in self._catalog.values():
            if doc.sha256_hash == file_hash:
                if metadata:
                    doc.metadata.update(metadata)
                if filename and doc.filename != filename:
                    doc.filename = filename
                self._save_catalog()
                return doc

        generated_id = doc_id or f"doc_{file_hash[:12]}"
        dest_filename = f"{generated_id}_{final_filename}"
        dest_path = self.docs_dir / dest_filename

        shutil.copy2(src_path, dest_path)

        suffix = Path(final_filename).suffix.lower().lstrip(".")
        mime_map = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "txt": "text/plain",
            "md": "text/markdown",
            "json": "application/json",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
        }
        mime_type = mime_map.get(suffix, "application/octet-stream")

        vault_doc = VaultDocument(
            doc_id=generated_id,
            filename=final_filename,
            original_path=str(src_path),
            vault_path=str(dest_path),
            file_size_bytes=dest_path.stat().st_size,
            sha256_hash=file_hash,
            mime_type=mime_type,
            metadata=metadata or {},
        )

        self._catalog[generated_id] = vault_doc
        self._save_catalog()
        logger.info(f"Stored original document '{final_filename}' in vault as '{generated_id}'")
        return vault_doc

    def store_parsed_markdown(
        self, doc_id: str, markdown_content: str, table_count: int = 0, chunk_count: int = 0
    ) -> None:
        """Stores parsed markdown and extraction statistics for a document."""
        doc = self.get_document(doc_id)
        if not doc:
            return

        parsed_file = self.parsed_dir / f"{doc_id}_parsed.md"
        parsed_file.write_text(markdown_content, encoding="utf-8")

        doc.parsed_markdown_path = str(parsed_file)
        doc.table_count = table_count
        doc.chunk_count = chunk_count
        self._save_catalog()

    def get_document(self, doc_id: str) -> VaultDocument | None:
        """Retrieves document record by doc_id."""
        return self._catalog.get(doc_id)

    def list_documents(self) -> list[VaultDocument]:
        """Returns all documents stored in the vault sorted by creation date."""
        return sorted(self._catalog.values(), key=lambda d: d.created_at, reverse=True)

    def get_parsed_content(self, doc_id: str) -> str | None:
        """Reads parsed markdown content for a document if available."""
        doc = self.get_document(doc_id)
        if doc and doc.parsed_markdown_path and Path(doc.parsed_markdown_path).exists():
            return Path(doc.parsed_markdown_path).read_text(encoding="utf-8")
        return None

    def build_citation(
        self, doc_id: str, chunk_id: str, snippet: str, page: int | None = None, score: float = 1.0
    ) -> dict[str, Any]:
        """Constructs a structured citation payload referencing the stored original document."""
        doc = self.get_document(doc_id)
        return {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "filename": doc.filename if doc else "unknown",
            "vault_path": doc.vault_path if doc else "",
            "file_size": doc.file_size_bytes if doc else 0,
            "mime_type": doc.mime_type if doc else "text/plain",
            "page": page,
            "snippet": snippet,
            "score": round(score, 4),
        }
