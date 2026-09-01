"""Document parser specialist subagent wrapping Docling, TableFormer, and HybridChunker."""

import os
from typing import Any

from config import WorkOSConfig
from workos_engine.agents.base import BaseSubagent
from workos_engine.tools.doc_tools import DocToolKit
from workos_engine.types import ExecutionResult, SubagentTask


class DocAgent(BaseSubagent):
    """Specialist subagent for parsing documents, extracting structured tables with TableFormer,

    extracting figures, and producing structure-aware chunks with Docling.
    """

    name: str = "doc_agent"
    description: str = (
        "Specialist subagent for multimodal document parsing (PDF, DOCX, PPTX, HTML, Markdown, Images), "
        "high-accuracy table extraction via TableFormer, figure extraction, and structure-aware chunking."
    )

    def __init__(
        self,
        config: WorkOSConfig | None = None,
        toolkit: DocToolKit | None = None,
    ):
        super().__init__(config=config)
        self.toolkit = toolkit or DocToolKit(config=self.config)

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a delegated document parsing, table extraction, figure extraction, or chunking instruction."""
        instruction = (task.instruction or "").lower().strip()
        ctx = dict(task.context or {})

        try:
            artifacts: list[str] = []
            if instruction in ("parse_document", "parse", "convert"):
                data = self.toolkit.parse_document(**ctx)
            elif instruction in ("extract_tables", "tables", "table_former"):
                data = self.toolkit.extract_tables(**ctx)
            elif instruction in ("extract_figures", "figures", "images"):
                data = self.toolkit.extract_figures(**ctx)
                if isinstance(data, list):
                    for fig in data:
                        img_path = fig.get("image_path")
                        if img_path and os.path.exists(img_path):
                            artifacts.append(img_path)
            elif instruction in ("structure_aware_chunk", "chunk", "chunk_document"):
                data = self.toolkit.structure_aware_chunk(**ctx)
            elif instruction in (
                "get_stored_document",
                "get_document",
                "fetch_document",
            ):
                data = self.toolkit.get_stored_document(**ctx)
            elif instruction in ("list_stored_documents", "list_documents"):
                data = self.toolkit.list_stored_documents()
            else:
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=False,
                    error=f"Unknown instruction: {task.instruction}",
                )

            return ExecutionResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=True,
                data=data,
                artifacts=artifacts,
            )
        except Exception as e:
            return ExecutionResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns JSON schema definitions of all tools provided by DocAgent for LLM planning."""
        return [
            {
                "name": "parse_document",
                "description": "Parse a document (PDF, DOCX, PPTX, HTML, Markdown, Images) into structured content, extracting tables and metadata using Docling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file to parse.",
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["markdown", "json", "text", "html", "doctags"],
                            "description": "Target output format for parsed content. Defaults to 'markdown'.",
                        },
                        "ocr": {
                            "type": "boolean",
                            "description": "Enable OCR for scanned documents or images. Defaults to false.",
                        },
                        "table_former": {
                            "type": "boolean",
                            "description": "Enable TableFormer for high-precision table structure recognition. Defaults to true.",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Optional custom identifier to store the document under in the session store.",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "extract_tables",
                "description": "Extract all structured tables from a document or stored document with TableFormer headers, row/column counts, and dataframes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file.",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID from a previously parsed document.",
                        },
                    },
                },
            },
            {
                "name": "extract_figures",
                "description": "Extract images, figures, charts, and diagrams from a document, with optional disk export.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file.",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID from a previously parsed document.",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Optional directory to save extracted figure image files.",
                        },
                    },
                },
            },
            {
                "name": "structure_aware_chunk",
                "description": "Chunk a document into coherent structure-aware chunks respecting headings, paragraphs, and tables via Docling HybridChunker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file.",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID from a previously parsed document.",
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum token threshold per chunk. Defaults to 512.",
                        },
                    },
                },
            },
            {
                "name": "get_stored_document",
                "description": "Retrieve content and metadata for a previously parsed document by its document ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Unique identifier of the stored document.",
                        },
                    },
                    "required": ["doc_id"],
                },
            },
            {
                "name": "list_stored_documents",
                "description": "List all parsed and stored documents available in the current session.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]
