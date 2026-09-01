"""Elasticsearch Parent-Child Hybrid RAG Specialist Subagent for WorkOS."""

import os
from typing import Any

from config import WorkOSConfig
from workos_engine.agents.base import BaseSubagent
from workos_engine.tools.rag_tools import RAGToolKit
from workos_engine.types import ExecutionResult, SubagentTask


class RAGAgent(BaseSubagent):
    """Specialist subagent for Elasticsearch parent-child hybrid RAG (RRF BM25 + dense vectors),

    grounded question answering with citations, and mathematical query complexity routing.
    """

    name: str = "rag_agent"
    description: str = (
        "Specialist subagent for high-scale Elasticsearch parent-child hybrid RAG (RRF fusion: BM25 + dense vectors), "
        "parent context resolution, grounded question answering with citations, and query complexity routing."
    )

    def __init__(
        self,
        config: WorkOSConfig | None = None,
        toolkit: RAGToolKit | None = None,
        model_client: Any | None = None,
    ):
        super().__init__(config=config, model_client=model_client)
        self.toolkit = toolkit or RAGToolKit(config=self.config)

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Dispatches an individual tool call for the RAGAgent."""
        t = tool_name.lower().strip()
        if t in ("rag_search", "search", "hybrid_search", "find"):
            return self.toolkit.rag_search(**args)
        elif t in (
            "rag_ingest_pdf",
            "ingest_pdf",
            "ingest",
            "ingest_document",
        ):
            return self.toolkit.rag_ingest_pdf(**args)
        elif t in ("rag_ask", "ask", "ask_question", "qa"):
            return self.toolkit.rag_ask(**args)
        elif t in ("rag_route_query", "route_query", "route", "classify"):
            return self.toolkit.rag_route_query(**args)
        else:
            raise ValueError(f"Unknown RAG tool: {tool_name}")

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a delegated RAG instruction or runs autonomous micro-ReAct loop."""
        instruction = (task.instruction or "").lower().strip()
        ctx = dict(task.context or {})

        mapped_tools = {
            "rag_search": "rag_search",
            "search": "rag_search",
            "hybrid_search": "rag_search",
            "find": "rag_search",
            "rag_ingest_pdf": "rag_ingest_pdf",
            "ingest_pdf": "rag_ingest_pdf",
            "ingest": "rag_ingest_pdf",
            "ingest_document": "rag_ingest_pdf",
            "rag_ask": "rag_ask",
            "ask": "rag_ask",
            "ask_question": "rag_ask",
            "qa": "rag_ask",
            "rag_route_query": "rag_route_query",
            "route_query": "rag_route_query",
            "route": "rag_route_query",
            "classify": "rag_route_query",
        }

        if instruction in mapped_tools:
            tool_name = mapped_tools[instruction]
            try:
                artifacts: list[str] = []
                data = self.execute_tool(tool_name, ctx)
                if (
                    isinstance(data, dict)
                    and "file_path" in data
                    and data["file_path"]
                    and os.path.exists(data["file_path"])
                ):
                    artifacts.append(data["file_path"])
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

        # Higher-level RAG mission -> run micro-ReAct loop
        return super().execute(task)

    def ingest_document(
        self,
        file_path: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Direct helper to ingest a document."""
        return self.toolkit.rag_ingest_pdf(file_path=file_path, doc_id=doc_id, metadata=metadata)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Direct helper to perform hybrid RRF search."""
        return self.toolkit.search(
            query=query, top_k=top_k, alpha=alpha, filter_metadata=filter_metadata
        )

    def ask_question(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> dict[str, Any]:
        """Direct helper to ask a question with grounded citations."""
        return self.toolkit.ask(query=query, top_k=top_k, alpha=alpha)

    def route_query(self, query: str) -> dict[str, Any]:
        """Direct helper to classify query complexity."""
        return self.toolkit.route_query(query=query)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns JSON schema definitions of all tools provided by RAGAgent for LLM planning."""
        return [
            {
                "name": "rag_search",
                "description": (
                    "Hybrid search combining BM25 lexical ranking and dense vector embeddings with Reciprocal Rank "
                    "Fusion (RRF) and parent section context resolution."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language or keyword search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of parent-child results to return. Defaults to 5.",
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Dense vs lexical weight balance (0.0 = pure BM25, 1.0 = pure dense vector). Defaults to 0.5.",
                        },
                        "filter_metadata": {
                            "type": "object",
                            "description": "Optional metadata key-value filters.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "rag_ingest_pdf",
                "description": (
                    "Ingest a PDF or document into the Elasticsearch parent-child hybrid index, splitting into "
                    "granular child chunks linked to broad parent sections with dense vector embeddings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document or PDF file to ingest.",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Optional custom document identifier.",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata dictionary to attach to all chunks.",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "rag_ask",
                "description": (
                    "Search the knowledge base and synthesize a factual answer strictly grounded in the retrieved "
                    "parent-child context, providing structured source citations."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question to answer using the knowledge base.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of retrieved chunks for context grounding. Defaults to 5.",
                        },
                        "alpha": {
                            "type": "number",
                            "description": "Hybrid search balance parameter. Defaults to 0.5.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "rag_route_query",
                "description": (
                    "Mathematical query complexity classifier that computes Shannon entropy, IDF specificity, "
                    "and reasoning indicators to route between DIRECT_SEARCH, HYBRID_RAG, and MULTI_HOP_REASONING."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to classify for complexity and optimal retrieval strategy.",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]
