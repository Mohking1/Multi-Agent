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
        model_client: Any | None = None,
    ):
        super().__init__(config=config, model_client=model_client)
        self.toolkit = toolkit or DocToolKit(config=self.config)

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Dispatches an individual tool call for the DocAgent."""
        t = tool_name.lower().strip()
        clean_args = dict(args)
        if t in ("parse_document", "parse", "convert"):
            if "file_path" not in clean_args:
                for alt_key in ("file_paths", "artifacts", "saved_paths", "saved_path", "path", "files"):
                    if alt_key in clean_args and clean_args[alt_key]:
                        clean_args["file_path"] = clean_args[alt_key]
                        break
            valid_keys = {"file_path", "output_format", "ocr", "table_former", "doc_id"}
            filtered_args = {k: v for k, v in clean_args.items() if k in valid_keys}
            return self.toolkit.parse_document(**filtered_args)
        elif t in ("extract_tables", "tables", "table_former"):
            return self.toolkit.extract_tables(**args)
        elif t in ("extract_figures", "figures", "images"):
            return self.toolkit.extract_figures(**args)
        elif t in ("structure_aware_chunk", "chunk", "chunk_document"):
            return self.toolkit.structure_aware_chunk(**args)
        elif t in ("get_stored_document", "get_document", "fetch_document"):
            return self.toolkit.get_stored_document(**args)
        elif t in ("list_stored_documents", "list_documents"):
            return self.toolkit.list_stored_documents()
        else:
            raise ValueError(f"Unknown doc tool: {tool_name}")

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a delegated document instruction."""
        instruction = (task.instruction or "").lower().strip()
        ctx = dict(task.context or {})

        mapped_tools = {
            "parse_document": "parse_document",
            "parse": "parse_document",
            "convert": "parse_document",
            "extract_tables": "extract_tables",
            "tables": "extract_tables",
            "table_former": "extract_tables",
            "extract_figures": "extract_figures",
            "figures": "extract_figures",
            "images": "extract_figures",
            "structure_aware_chunk": "structure_aware_chunk",
            "chunk": "structure_aware_chunk",
            "chunk_document": "structure_aware_chunk",
            "get_stored_document": "get_stored_document",
            "get_document": "get_stored_document",
            "fetch_document": "get_stored_document",
            "list_stored_documents": "list_stored_documents",
            "list_documents": "list_stored_documents",
        }

        # Normalize flexible file path arguments from upstream steps
        if "file_path" not in ctx:
            for alt_key in ("file_paths", "artifacts", "saved_paths", "saved_path", "path", "files"):
                if alt_key in ctx and ctx[alt_key]:
                    ctx["file_path"] = ctx[alt_key]
                    break

        if instruction in mapped_tools:
            tool_name = mapped_tools[instruction]
            try:
                artifacts: list[str] = []
                data = self.execute_tool(tool_name, ctx)
                if isinstance(data, list):
                    for item in data:
                        if (
                            isinstance(item, dict)
                            and "image_path" in item
                            and os.path.exists(item["image_path"])
                        ):
                            artifacts.append(item["image_path"])
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

        return ExecutionResult(
            task_id=task.task_id,
            agent_name=self.name,
            success=False,
            error=f"Unknown instruction: {task.instruction}",
        )

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns JSON schema definitions of all tools provided by DocAgent for LLM planning."""
        return [
            {
                "name": "parse_document",
                "description": "Parse a document or list of documents (PDF, DOCX, PPTX, HTML, Markdown, Images) into structured content, extracting tables and metadata using Docling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "description": "Path to the document file or list of file paths to parse.",
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
