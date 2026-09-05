"""Docling, TableFormer, and HybridChunker Document Parsing ToolKit for WorkOS."""

import hashlib
import json
import logging
import os
import time
from typing import Any

try:
    from docling.chunking import HybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    HAS_DOCLING = True
except ImportError:
    HybridChunker = None  # type: ignore
    InputFormat = None  # type: ignore
    PdfPipelineOptions = None  # type: ignore
    TableFormerMode = None  # type: ignore
    AcceleratorOptions = None  # type: ignore
    AcceleratorDevice = None  # type: ignore
    DocumentConverter = None  # type: ignore
    PdfFormatOption = None  # type: ignore
    HAS_DOCLING = False

from config import WorkOSConfig, get_config
from workos_engine.vault import DocumentVault

logger = logging.getLogger(__name__)


class DocToolKit:
    """ToolKit providing Docling document conversion, TableFormer extraction, and HybridChunker operations."""

    def __init__(self, config: WorkOSConfig | None = None, vault: DocumentVault | None = None):
        self.config = config or get_config()
        self.vault = vault or DocumentVault(vault_dir=self.config.vault_dir)
        self._documents: dict[str, dict[str, Any]] = {}

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a deterministic document identifier based on file path and filename."""
        base = os.path.basename(file_path)
        path_hash = hashlib.sha256(os.path.abspath(file_path).encode("utf-8")).hexdigest()[:8]
        clean_base = "".join(c if c.isalnum() else "_" for c in os.path.splitext(base)[0])
        return f"doc_{clean_base}_{path_hash}"

    def _get_converter(self, ocr: bool = False, table_former: bool = True) -> DocumentConverter:
        """Instantiates a Docling DocumentConverter with configured pipeline options."""
        if not HAS_DOCLING or DocumentConverter is None or PdfPipelineOptions is None:
            raise RuntimeError(
                "Docling is not installed. Please install docling to use document parsing tools."
            )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = ocr
        pipeline_options.do_table_structure = table_former
        if table_former:
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        if AcceleratorOptions is not None and AcceleratorDevice is not None:
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.CPU
            )

        format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        return DocumentConverter(format_options=format_options)

    def parse_document(
        self,
        file_path: str | list[str],
        output_format: str = "markdown",
        ocr: bool = False,
        table_former: bool = True,
        doc_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Parses a document file or list of files (PDF, DOCX, PPTX, HTML, Markdown, Images) into structured content.

        Args:
            file_path: Path to the document file or list of file paths.
            output_format: Desired output format ('markdown', 'json', 'text', 'html', 'doctags').
            ocr: Whether to run OCR for scanned pages/images.
            table_former: Whether to use TableFormer for high-fidelity table extraction.
            doc_id: Optional custom identifier to store the document under.

        Returns:
            Dictionary containing document ID, metadata, tables count, figures count, and parsed content.
        """
        if isinstance(file_path, list):
            if not file_path:
                return {
                    "doc_id": "empty_batch",
                    "file_path": "",
                    "file_paths": [],
                    "content": "No files provided to parse.",
                    "num_documents": 0,
                    "documents": [],
                }
            if len(file_path) == 1:
                return self.parse_document(
                    file_path=file_path[0],
                    output_format=output_format,
                    ocr=ocr,
                    table_former=table_former,
                    doc_id=doc_id,
                )
            # Batch parse multiple documents
            doc_results = []
            combined_content_parts = []
            total_tables = 0
            total_figures = 0
            for p in file_path:
                single_res = self.parse_document(
                    file_path=p,
                    output_format=output_format,
                    ocr=ocr,
                    table_former=table_former,
                )
                doc_results.append(single_res)
                total_tables += single_res.get("tables_count", 0)
                total_figures += single_res.get("figures_count", 0)
                fname = os.path.basename(p)
                combined_content_parts.append(
                    f"### Document: {fname}\n\n{single_res.get('content', '')}"
                )

            batch_id = f"batch_{int(time.time())}_{len(file_path)}"
            return {
                "doc_id": batch_id,
                "file_path": file_path[0] if file_path else "",
                "file_paths": file_path,
                "num_documents": len(doc_results),
                "format": output_format,
                "content": "\n\n---\n\n".join(combined_content_parts),
                "tables_count": total_tables,
                "figures_count": total_figures,
                "documents": doc_results,
            }

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        abs_path = os.path.abspath(file_path)
        converter = self._get_converter(ocr=ocr, table_former=table_former)
        res = converter.convert(abs_path)
        doc = res.document

        fmt = (output_format or "markdown").lower().strip()
        if fmt in ("markdown", "md"):
            content = doc.export_to_markdown()
        elif fmt == "json":
            content = doc.export_to_dict()
        elif fmt in ("text", "txt"):
            content = doc.export_to_text()
        elif fmt == "html":
            content = doc.export_to_html()
        elif fmt == "doctags":
            if hasattr(doc, "export_to_document_tokens"):
                content = doc.export_to_document_tokens()
            elif hasattr(doc, "export_to_doctags"):
                content = doc.export_to_doctags()
            else:
                content = doc.export_to_markdown()
        else:
            content = doc.export_to_markdown()

        tables_count = len(getattr(doc, "tables", []) or [])
        figures_count = len(getattr(doc, "pictures", []) or [])
        pages = getattr(doc, "pages", {}) or {}
        num_pages = len(pages) if isinstance(pages, (dict, list)) and len(pages) > 0 else 1

        stored_id = doc_id or self._generate_doc_id(abs_path)
        file_name = os.path.basename(abs_path)

        stored_record = {
            "doc_id": stored_id,
            "file_path": abs_path,
            "file_name": file_name,
            "format": fmt,
            "content": content,
            "num_pages": num_pages,
            "tables_count": tables_count,
            "figures_count": figures_count,
            "metadata": {
                "file_size": os.path.getsize(abs_path),
                "ocr_enabled": ocr,
                "table_former_enabled": table_former,
            },
            "created_at": time.time(),
            "_docling_doc": doc,
        }
        self._documents[stored_id] = stored_record

        # Store original document and parsed markdown into permanent Vault
        try:
            vault_doc = self.vault.store_document(
                abs_path, doc_id=stored_id, metadata=stored_record["metadata"]
            )
            self.vault.store_parsed_markdown(
                stored_id,
                content if isinstance(content, str) else json.dumps(content),
                table_count=tables_count,
            )
            vault_path = vault_doc.vault_path
        except Exception as e:
            logger.warning(f"Could not persist document '{stored_id}' to vault: {e}")
            vault_path = abs_path

        # Return public response without internal docling document pointer
        return {
            "doc_id": stored_id,
            "file_path": abs_path,
            "vault_path": vault_path,
            "file_name": file_name,
            "format": fmt,
            "content": content,
            "num_pages": num_pages,
            "tables_count": tables_count,
            "figures_count": figures_count,
            "metadata": stored_record["metadata"],
        }

    def _resolve_document(
        self, file_path: str | None = None, doc_id: str | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Resolves a Docling Document object and its stored record from doc_id or file_path."""
        if doc_id:
            if doc_id not in self._documents:
                raise KeyError(f"Document with ID '{doc_id}' not found in store.")
            record = self._documents[doc_id]
            return record.get("_docling_doc"), record

        if file_path:
            abs_path = os.path.abspath(file_path)
            for record in self._documents.values():
                if record.get("file_path") == abs_path:
                    return record.get("_docling_doc"), record

            # If not yet parsed, parse it now
            parsed = self.parse_document(file_path=file_path)
            record = self._documents[parsed["doc_id"]]
            return record.get("_docling_doc"), record

        raise ValueError("Either file_path or doc_id must be provided.")

    def extract_tables(
        self,
        file_path: str | None = None,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extracts structured tables with headers, row/column counts, markdown, HTML, and dataframes.

        Args:
            file_path: Optional path to the document file.
            doc_id: Optional document identifier from a previous parse.

        Returns:
            List of structured table dictionaries.
        """
        doc, _ = self._resolve_document(file_path=file_path, doc_id=doc_id)
        tables = getattr(doc, "tables", []) or []

        extracted: list[dict[str, Any]] = []
        for idx, table in enumerate(tables):
            # Markdown export
            try:
                table_md = table.export_to_markdown(doc=doc)
            except TypeError:
                table_md = table.export_to_markdown()
            except Exception:
                table_md = ""

            # HTML export
            try:
                table_html = table.export_to_html(doc=doc)
            except TypeError:
                table_html = table.export_to_html()
            except Exception:
                table_html = ""

            # Dataframe export
            try:
                df = table.export_to_dataframe(doc=doc)
            except TypeError:
                df = table.export_to_dataframe()
            except Exception:
                df = None

            if df is not None:
                if hasattr(df, "shape") and isinstance(df.shape, tuple) and len(df.shape) >= 2:
                    num_rows = int(df.shape[0])
                    num_cols = int(df.shape[1])
                elif hasattr(df, "__len__"):
                    try:
                        num_rows = len(df)
                    except Exception:
                        num_rows = 0
                    num_cols = len(getattr(df, "columns", [])) if hasattr(df, "columns") else 0
                else:
                    num_rows = 0
                    num_cols = 0

                if hasattr(df, "to_dict"):
                    try:
                        df_data = df.to_dict(orient="records")
                    except Exception:
                        df_data = []
                else:
                    df_data = []
            else:
                df_data = []
                num_rows = 0
                num_cols = 0

            # Caption
            caption = ""
            if hasattr(table, "caption_text"):
                try:
                    caption = table.caption_text(doc=doc) or ""
                except TypeError:
                    caption = table.caption_text() or ""
                except Exception:
                    caption = ""
            elif hasattr(table, "caption"):
                caption = str(table.caption or "")

            extracted.append(
                {
                    "table_index": idx,
                    "caption": caption,
                    "markdown": table_md,
                    "html": table_html,
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "dataframe": df_data,
                }
            )

        return extracted

    def extract_figures(
        self,
        file_path: str | None = None,
        doc_id: str | None = None,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extracts pictures, charts, diagrams, and figures from a document.

        Args:
            file_path: Optional path to the document file.
            doc_id: Optional document identifier from a previous parse.
            output_dir: Optional local directory to save extracted figure images.

        Returns:
            List of figure metadata dictionaries, including saved image paths if output_dir is given.
        """
        doc, _ = self._resolve_document(file_path=file_path, doc_id=doc_id)
        pictures = getattr(doc, "pictures", []) or []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        extracted: list[dict[str, Any]] = []
        for idx, pic in enumerate(pictures):
            caption = ""
            if hasattr(pic, "caption_text"):
                try:
                    caption = pic.caption_text(doc=doc) or ""
                except TypeError:
                    caption = pic.caption_text() or ""
                except Exception:
                    caption = ""

            annotations = list(getattr(pic, "annotations", []) or [])
            fig_data: dict[str, Any] = {
                "figure_index": idx,
                "caption": caption,
                "annotations": annotations,
            }

            if output_dir and hasattr(pic, "get_image"):
                try:
                    img = pic.get_image(doc=doc)
                except TypeError:
                    img = pic.get_image()
                except Exception:
                    img = None

                if img is not None and hasattr(img, "save"):
                    img_path = os.path.join(output_dir, f"figure_{idx}.png")
                    img.save(img_path)
                    fig_data["image_path"] = img_path

            extracted.append(fig_data)

        return extracted

    def structure_aware_chunk(
        self,
        file_path: str | None = None,
        doc_id: str | None = None,
        max_tokens: int = 512,
    ) -> list[dict[str, Any]]:
        """Splits document content into structure-aware chunks respecting sections, headings, and tables.

        Args:
            file_path: Optional path to the document file.
            doc_id: Optional document identifier from a previous parse.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of structure-aware chunk dictionaries with text, headings, and item metadata.
        """
        doc, _ = self._resolve_document(file_path=file_path, doc_id=doc_id)
        chunker = HybridChunker(max_tokens=max_tokens)
        chunks_iter = chunker.chunk(doc)

        result_chunks: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks_iter):
            chunk_text = getattr(chunk, "text", str(chunk))
            meta = getattr(chunk, "meta", None)
            headings = list(getattr(meta, "headings", []) or [])
            doc_items = [str(item) for item in (getattr(meta, "doc_items", []) or [])]

            result_chunks.append(
                {
                    "chunk_index": idx,
                    "text": chunk_text,
                    "headings": headings,
                    "doc_items": doc_items,
                    "token_count": len(chunk_text.split()),
                }
            )

        return result_chunks

    def get_stored_document(self, doc_id: str) -> dict[str, Any]:
        """Retrieves parsed document content and metadata by document ID."""
        if doc_id not in self._documents:
            raise KeyError(f"Document with ID '{doc_id}' not found in store.")
        rec = self._documents[doc_id]
        return {
            "doc_id": rec["doc_id"],
            "file_path": rec["file_path"],
            "file_name": rec["file_name"],
            "format": rec["format"],
            "content": rec["content"],
            "num_pages": rec["num_pages"],
            "tables_count": rec["tables_count"],
            "figures_count": rec["figures_count"],
            "metadata": rec["metadata"],
            "created_at": rec.get("created_at"),
        }

    def list_stored_documents(self) -> list[dict[str, Any]]:
        """Lists metadata summaries for all stored documents in this session."""
        return [
            {
                "doc_id": doc_id,
                "file_path": rec["file_path"],
                "file_name": rec["file_name"],
                "format": rec["format"],
                "num_pages": rec["num_pages"],
                "tables_count": rec["tables_count"],
                "figures_count": rec["figures_count"],
                "created_at": rec.get("created_at"),
            }
            for doc_id, rec in self._documents.items()
        ]
