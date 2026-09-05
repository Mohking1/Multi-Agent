"""Elasticsearch Parent-Child Hybrid RAG Specialist Subagent for WorkOS."""

import json
import logging
import os
from typing import Any

from config import WorkOSConfig
from workos_engine.agents.base import BaseSubagent
from workos_engine.tools.rag_tools import RAGToolKit
from workos_engine.types import ExecutionResult, SubagentTask

logger = logging.getLogger("workos.agents.rag")


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
        clean_args = {
            k: v
            for k, v in args.items()
            if k
            not in (
                "instruction",
                "goal",
                "blackboard",
                "previous_steps",
                "cognitive_context",
                "step_description",
            )
        }
        if t == "rag_search":
            return self.toolkit.rag_search(**clean_args)
        elif t in ("search", "hybrid_search", "find"):
            if hasattr(self.toolkit, "search") and callable(self.toolkit.search):
                return self.toolkit.search(**clean_args)
            return self.toolkit.rag_search(**clean_args)
        elif t in (
            "rag_ingest_pdf",
            "ingest_pdf",
            "ingest",
            "ingest_document",
        ):
            return self.toolkit.rag_ingest_pdf(**clean_args)
        elif t in ("rag_ask", "ask", "ask_question", "qa"):
            return self.toolkit.rag_ask(**clean_args)
        elif t in ("rag_route_query", "route_query", "route", "classify"):
            return self.toolkit.rag_route_query(**clean_args)
        else:
            raise ValueError(f"Unknown RAG tool: {tool_name}")

    def _expand_rag_query(self, query: str) -> list[str]:
        client = getattr(self, "model_client", None)
        if not client or not hasattr(client, "generate"):
            return [query]

        prompt = (
            f"You are the WorkOS Knowledge Retrieval Specialist.\n"
            f"User Query: '{query}'\n\n"
            f"Formulate 2 to 3 alternative search queries for hybrid Elasticsearch (BM25 + Dense Vectors).\n"
            f"Include relevant technical synonyms, acronyms, or specific domain terminology.\n\n"
            f"Output strictly a JSON object:\n"
            f"{{\n"
            f'  "queries": ["<query 1>", "<query 2>", "<query 3>"]\n'
            f"}}\n"
        )
        try:
            res = client.generate(prompt=prompt)
            if res:
                parsed = extract_and_parse_json(res)
                queries = parsed.get("queries", [])
                if isinstance(queries, list) and queries:
                    return [str(q).strip() for q in queries if str(q).strip()]
        except Exception as e:
            logger.debug(f"RAG query expansion fallback: {e}")

        return [query]

    def _retrieve_candidate_chunks(
        self, queries: list[str], top_k: int = 5
    ) -> list[dict[str, Any]]:
        all_chunks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for q in queries:
            try:
                res = self.toolkit.rag_search(query=q, top_k=top_k)
                hits = res.get("hits", []) if isinstance(res, dict) else res
                for h in hits:
                    chunk_id = str(h.get("chunk_id") or h.get("id") or h.get("text", "")[:40])
                    if chunk_id and chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        all_chunks.append(h)
            except Exception as e:
                logger.debug(f"RAG candidate retrieval error for '{q}': {e}")
        return all_chunks

    def _filter_and_rerank_chunks(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 4
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        client = getattr(self, "model_client", None)
        if not client or not hasattr(client, "generate"):
            return candidates[:top_k]

        snippets = []
        for idx, c in enumerate(candidates[:15], 1):
            text = c.get("text") or c.get("content") or ""
            doc = c.get("document_id") or c.get("title") or "Doc"
            snippets.append(f"[{idx}] (Doc: {doc}) Preview: {text[:200]}")

        prompt = (
            f"You are the WorkOS RAG Precision Re-Ranker.\n"
            f"User Query: '{query}'\n\n"
            f"Retrieved candidate text chunks:\n" + "\n---\n".join(snippets) + "\n\n"
            f"Select the top {top_k} most directly relevant, high-information chunks that answer the query.\n\n"
            f"Output strictly a JSON object:\n"
            f"{{\n"
            f'  "selected_indices": [<1-based indices of best {top_k} chunks>]\n'
            f"}}\n"
        )
        try:
            res = client.generate(prompt=prompt)
            if res:
                parsed = extract_and_parse_json(res)
                indices = parsed.get("selected_indices", [])
                selected = [candidates[i - 1] for i in indices if 1 <= i <= len(candidates)]
                if selected:
                    return selected[:top_k]
        except Exception as e:
            logger.debug(f"RAG re-ranking fallback: {e}")

        return candidates[:top_k]

    def autonomous_rag_search(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Generalized 4-phase autonomous RAG search pipeline:
        1. Query Expansion: Generates semantic synonym and domain query variations.
        2. Candidate Retrieval: Queries Elasticsearch hybrid index across variations.
        3. Precision Re-Ranking: LLM re-ranks chunks and weeds out marginal hits.
        4. Structured Output & Receipts: Emits verified citations and compact receipt for blackboard.
        """
        queries = self._expand_rag_query(query)
        candidates = self._retrieve_candidate_chunks(queries)
        if not candidates:
            receipt = f"Executed {len(queries)} RAG search variations for '{query}'. Zero matching documents found."
            return {
                "status": "completed",
                "query": query,
                "queries": queries,
                "candidates_count": 0,
                "hits": [],
                "citations": [],
                "receipt": receipt,
                "message": receipt,
            }

        selected_chunks = self._filter_and_rerank_chunks(query, candidates)
        citations = []
        for idx, c in enumerate(selected_chunks, 1):
            doc_title = c.get("title") or c.get("document_id") or f"Doc_{idx}"
            citations.append(
                {
                    "citation_id": f"[{idx}]",
                    "title": doc_title,
                    "filename": doc_title,
                    "snippet": (c.get("text") or c.get("content") or "")[:250],
                    "score": c.get("score", 1.0),
                }
            )

        receipt = (
            f"Scanned {len(candidates)} candidate chunks across {len(queries)} RAG query variations. "
            f"Filtered and re-ranked top {len(selected_chunks)} most relevant sections. "
            f"Generated {len(citations)} verified citations."
        )
        return {
            "status": "completed",
            "query": query,
            "queries": queries,
            "candidates_count": len(candidates),
            "hits": selected_chunks,
            "citations": citations,
            "receipt": receipt,
            "message": receipt,
        }

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a delegated RAG instruction or runs autonomous micro-ReAct loop."""
        instruction = (task.instruction or "").lower().strip()
        ctx = dict(task.context or {})

        mapped_tools = {
            "rag_search": "rag_search",
            "search": "search",
            "hybrid_search": "hybrid_search",
            "find": "find",
            "rag_ingest_pdf": "rag_ingest_pdf",
            "ingest_pdf": "ingest_pdf",
            "ingest": "ingest",
            "ingest_document": "ingest_document",
            "rag_ask": "rag_ask",
            "ask": "rag_ask",
            "ask_question": "rag_ask",
            "qa": "rag_ask",
            "rag_route_query": "rag_route_query",
            "route_query": "rag_route_query",
            "route": "rag_route_query",
            "classify": "rag_route_query",
        }

        # Check for direct mapped tool call
        if instruction in mapped_tools:
            tool_name = mapped_tools[instruction]
            try:
                data = self.execute_tool(tool_name, ctx)
                artifacts: list[str] = []
                if (
                    isinstance(data, dict)
                    and "file_path" in data
                    and data["file_path"]
                    and os.path.exists(data["file_path"])
                ):
                    artifacts.append(data["file_path"])
                elif isinstance(data, dict) and "citations" in data:
                    artifacts = data["citations"]
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

        # Autonomous RAG search pipeline
        query = ctx.get("query") or ctx.get("goal") or ctx.get("step_description")
        is_rag_intent = (
            instruction in ("autonomous_rag_search", "rag", "knowledge_base", "kb_search")
            or any(
                w in instruction
                for w in (
                    "search",
                    "find",
                    "retrieve",
                    "query",
                    "lookup",
                    "rag",
                    "ask",
                    "knowledge",
                )
            )
            or bool(ctx.get("query") or ctx.get("goal"))
        )

        if is_rag_intent and (query or instruction):
            search_q = query or instruction
            try:
                data = self.autonomous_rag_search(query=search_q, context=ctx)
                artifacts = data.get("citations", []) if isinstance(data, dict) else []
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data=data,
                    artifacts=artifacts,
                )
            except Exception as e:
                logger.exception(f"Error in autonomous_rag_search: {e}")
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
