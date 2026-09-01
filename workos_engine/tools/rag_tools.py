"""Elasticsearch Parent-Child Hybrid RAG ToolKit for WorkOS."""

import hashlib
import logging
import math
import os
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

from config import WorkOSConfig, get_config
from workos_engine.tools.doc_tools import DocToolKit


class RAGToolKit:
    """ToolKit providing Elasticsearch parent-child hybrid retrieval (BM25 + dense vectors with RRF fusion),

    grounded citation QA, and mathematical query complexity routing.
    """

    STOP_WORDS = {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }

    def __init__(
        self,
        config: WorkOSConfig | None = None,
        es_client: Any | None = None,
        embedder: Any | None = None,
        doc_toolkit: DocToolKit | None = None,
        model_client: Any | None = None,
        vault: Any | None = None,
    ):
        self.config = config or get_config()
        self.es_url = self.config.elasticsearch_url
        self.index_name = self.config.rag_index_name
        self.parent_index_name = f"{self.index_name}_parents"
        self.embedder = embedder
        self.doc_toolkit = doc_toolkit
        if vault is not None:
            self.vault = vault
        elif doc_toolkit is not None and hasattr(doc_toolkit, "vault"):
            self.vault = doc_toolkit.vault
        else:
            from workos_engine.vault import DocumentVault

            self.vault = DocumentVault()

        if model_client is not None:
            self.model_client = model_client
        else:
            self.model_client = self._init_model_client()

        # Internal in-memory fallback stores for local testing / disconnected environments
        self._in_memory_parents: dict[str, dict[str, Any]] = {}
        self._in_memory_children: list[dict[str, Any]] = []

        if es_client is not None:
            self.es = es_client
        else:
            self.es = self._init_elasticsearch()

    def _init_model_client(self) -> Any:
        """Initializes OllamaClient for RAG embeddings and generation."""
        try:
            from workos_engine.llm_client import OllamaClient

            return OllamaClient(
                base_url=self.config.ollama_base_url,
                default_model=self.config.model_name,
                embedding_model=self.config.embedding_model,
            )
        except Exception:
            return None

    def _init_elasticsearch(self) -> Any | None:
        """Initializes Elasticsearch client if available and reachable."""
        if Elasticsearch is None:
            return None
        try:
            kwargs: dict[str, Any] = {
                "request_timeout": 1.0,
                "max_retries": 1,
                "retry_on_timeout": False,
            }
            if getattr(self.config, "elasticsearch_api_key", None):
                kwargs["api_key"] = self.config.elasticsearch_api_key
            elif getattr(self.config, "elasticsearch_password", None):
                kwargs["basic_auth"] = (
                    self.config.elasticsearch_username,
                    self.config.elasticsearch_password,
                )

            client = Elasticsearch(self.es_url, **kwargs)
            if client.ping():
                return client
        except Exception:
            pass
        return None

    def _get_embedding(self, text: str) -> list[float]:
        """Computes dense vector embedding for text using Ollama or deterministic projection."""
        if callable(self.embedder):
            return self.embedder(text)

        if self.model_client is not None and hasattr(self.model_client, "embed"):
            try:
                embeds = self.model_client.embed(text)
                if embeds and len(embeds) > 0:
                    return embeds[0]
            except Exception as e:
                logger.debug(f"Ollama embedding failed: {e}")

        # High-dimensional deterministic projection (768 dims)
        dims = 768
        vec = [0.0] * dims
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for pos, token in enumerate(tokens):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx1 = h % dims
            idx2 = (h >> 16) % dims
            idx3 = (h >> 32) % dims
            weight = 1.0 / (1.0 + 0.1 * pos)
            vec[idx1] += weight * 1.0
            vec[idx2] += weight * 0.5
            vec[idx3] += weight * 0.25

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec

    def route_query(self, query: str) -> dict[str, Any]:
        """Mathematical fast-probe query complexity classifier.

        Computes Shannon entropy, IDF specificity proxy, token density, and reasoning indicators
        to route query between DIRECT_SEARCH, HYBRID_RAG, and MULTI_HOP_REASONING.
        """
        raw_query = query or ""
        clean_text = raw_query.strip()
        tokens = re.findall(r"\w+", clean_text.lower())
        token_count = len(tokens)

        if token_count == 0:
            return {
                "query": query,
                "recommended_strategy": "DIRECT_SEARCH",
                "complexity_score": 0.0,
                "entropy": 0.0,
                "specificity_score": 0.0,
                "metrics": {
                    "token_count": 0,
                    "rare_token_ratio": 0.0,
                    "stop_word_ratio": 0.0,
                    "entropy": 0.0,
                    "specificity": 0.0,
                    "margin": 0.5,
                },
            }

        # 1. Stop words vs rare/domain tokens
        stop_count = sum(1 for t in tokens if t in self.STOP_WORDS)
        stop_word_ratio = stop_count / token_count
        rare_tokens = [
            t
            for t in tokens
            if t not in self.STOP_WORDS and (len(t) >= 4 or any(c.isdigit() for c in t))
        ]
        rare_token_ratio = len(rare_tokens) / token_count

        # 2. Exact identifiers check (e.g. INV-1234, #101, UUID, uppercase acronyms)
        has_identifier = bool(re.search(r"#[A-Za-z0-9_-]+|\b[A-Z]{2,}-\d+\b|\b\d{4,}\b", raw_query))

        # 3. Shannon Entropy of token distribution
        counts = Counter(tokens)
        entropy = -sum(
            (cnt / token_count) * math.log2(cnt / token_count) for cnt in counts.values()
        )

        # 4. Multi-hop and complex reasoning keywords
        reasoning_keywords = {
            "compare",
            "comparison",
            "difference",
            "versus",
            "vs",
            "analyze",
            "analysis",
            "why",
            "synthesize",
            "synthesis",
            "evaluate",
            "evaluation",
            "trend",
            "trends",
            "breakdown",
            "across",
            "relationship",
            "impact",
            "correlate",
            "correlation",
            "bottlenecks",
            "factors",
            "root",
            "cause",
        }
        reasoning_matches = [t for t in tokens if t in reasoning_keywords]
        reasoning_weight = min(1.0, len(reasoning_matches) * 0.35)

        # 5. Specificity Score [0.0, 1.0]
        specificity = min(
            1.0,
            max(
                0.0,
                (0.55 * rare_token_ratio)
                + (0.35 if has_identifier else 0.0)
                + (0.20 * (1.0 - stop_word_ratio)),
            ),
        )

        # 6. Complexity Score [0.0, 1.0]
        length_factor = min(1.0, token_count / 22.0)
        entropy_factor = min(1.0, entropy / 4.0)
        complexity = min(
            1.0,
            max(
                0.0,
                (0.35 * length_factor)
                + (0.35 * reasoning_weight)
                + (0.20 * entropy_factor)
                + (0.10 * (1.0 - specificity)),
            ),
        )

        # 7. Routing Strategy Selection
        if has_identifier and token_count <= 8:
            strategy = "DIRECT_SEARCH"
        elif reasoning_weight >= 0.35 or complexity > 0.45 or token_count > 18:
            strategy = "MULTI_HOP_REASONING"
        elif specificity > 0.65 and token_count <= 6:
            strategy = "DIRECT_SEARCH"
        else:
            strategy = "HYBRID_RAG"

        return {
            "query": query,
            "recommended_strategy": strategy,
            "complexity_score": round(complexity, 3),
            "entropy": round(entropy, 3),
            "specificity_score": round(specificity, 3),
            "metrics": {
                "token_count": token_count,
                "rare_token_ratio": round(rare_token_ratio, 3),
                "stop_word_ratio": round(stop_word_ratio, 3),
                "entropy": round(entropy, 3),
                "specificity": round(specificity, 3),
                "margin": round(abs(complexity - 0.5), 3),
            },
        }

    def rag_route_query(self, query: str) -> dict[str, Any]:
        """Alias for route_query."""
        return self.route_query(query)

    def create_parent_child_chunks(
        self,
        text: str,
        doc_id: str,
        parent_chunk_size: int = 600,
        child_chunk_size: int = 150,
        child_overlap: int = 30,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Splits raw text into broad parent sections and granular child chunks linked by parent_id."""
        parents: list[dict[str, Any]] = []
        children: list[dict[str, Any]] = []

        # Split into parent paragraphs/sections
        raw_sections = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
        if not raw_sections:
            raw_sections = [text.strip()] if text.strip() else []

        parent_idx = 0
        current_parent_text = ""

        def flush_parent(p_text: str, p_id: str):
            parents.append(
                {
                    "parent_id": p_id,
                    "doc_id": doc_id,
                    "text": p_text,
                    "parent_index": parent_idx,
                }
            )
            words = p_text.split()
            if not words:
                return

            c_idx = 0
            start = 0
            while start < len(words):
                end = min(len(words), start + child_chunk_size)
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                child_id = f"{p_id}_c{c_idx}"
                children.append(
                    {
                        "chunk_id": child_id,
                        "doc_id": doc_id,
                        "parent_id": p_id,
                        "text": chunk_text,
                        "chunk_index": c_idx,
                        "vector": self._get_embedding(chunk_text),
                    }
                )
                c_idx += 1
                if end >= len(words):
                    break
                start += max(1, child_chunk_size - child_overlap)

        for sec in raw_sections:
            if len(current_parent_text) + len(sec) < parent_chunk_size:
                current_parent_text = f"{current_parent_text}\n\n{sec}".strip()
            else:
                if current_parent_text:
                    p_id = f"{doc_id}_p{parent_idx}"
                    flush_parent(current_parent_text, p_id)
                    parent_idx += 1
                current_parent_text = sec

        if current_parent_text:
            p_id = f"{doc_id}_p{parent_idx}"
            flush_parent(current_parent_text, p_id)

        return parents, children

    def rag_ingest_pdf(
        self,
        file_path: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingests a PDF or text document into the Elasticsearch parent-child hybrid index.

        Args:
            file_path: Path to the document file.
            doc_id: Optional custom identifier.
            metadata: Optional metadata dictionary.

        Returns:
            Dictionary reporting ingestion status and chunk statistics.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        abs_path = os.path.abspath(file_path)
        base = os.path.basename(abs_path)
        if not doc_id:
            h = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:8]
            clean_name = "".join(c if c.isalnum() else "_" for c in os.path.splitext(base)[0])
            doc_id = f"doc_{clean_name}_{h}"

        text_content = ""
        # 1. Try DocToolKit if file is PDF/DOCX/HTML
        if self.doc_toolkit is not None:
            try:
                parsed = self.doc_toolkit.parse_document(abs_path, output_format="markdown")
                text_content = parsed.get("content", "")
            except Exception:
                pass

        # 2. Fallback direct text reading
        if not text_content:
            try:
                with open(abs_path, encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
            except Exception as e:
                raise RuntimeError(f"Failed to read document text: {e}")

        # 3. Create parent-child chunks
        parents, children = self.create_parent_child_chunks(text_content, doc_id=doc_id)

        meta = metadata or {}

        # 4. Ingest into Elasticsearch if available
        if self.es is not None:
            try:
                if hasattr(self.es, "indices") and hasattr(self.es.indices, "exists"):
                    if not self.es.indices.exists(index=self.index_name):
                        self.es.indices.create(
                            index=self.index_name,
                            body={
                                "mappings": {
                                    "properties": {
                                        "chunk_id": {"type": "keyword"},
                                        "doc_id": {"type": "keyword"},
                                        "parent_id": {"type": "keyword"},
                                        "text": {"type": "text"},
                                        "metadata": {"type": "object", "enabled": True},
                                        "vector": {
                                            "type": "dense_vector",
                                            "dims": 768,
                                            "index": True,
                                            "similarity": "cosine",
                                        },
                                    }
                                }
                            },
                        )
                    if not self.es.indices.exists(index=self.parent_index_name):
                        self.es.indices.create(
                            index=self.parent_index_name,
                            body={
                                "mappings": {
                                    "properties": {
                                        "parent_id": {"type": "keyword"},
                                        "doc_id": {"type": "keyword"},
                                        "text": {"type": "text"},
                                        "metadata": {"type": "object", "enabled": True},
                                    }
                                }
                            },
                        )

                for p in parents:
                    self.es.index(
                        index=self.parent_index_name,
                        id=p["parent_id"],
                        document={**p, "metadata": meta, "file_path": abs_path},
                    )

                for c in children:
                    self.es.index(
                        index=self.index_name,
                        id=c["chunk_id"],
                        document={**c, "metadata": meta, "file_path": abs_path},
                    )
            except Exception:
                pass

        # 5. Populate in-memory fallback store
        for p in parents:
            self._in_memory_parents[p["parent_id"]] = {
                **p,
                "metadata": meta,
                "file_path": abs_path,
            }

        for c in children:
            self._in_memory_children.append(
                {
                    **c,
                    "metadata": meta,
                    "file_path": abs_path,
                }
            )

        return {
            "doc_id": doc_id,
            "file_path": abs_path,
            "parent_chunks_count": len(parents),
            "child_chunks_count": len(children),
            "status": "indexed",
            "metadata": meta,
        }

    def ingest_document(
        self,
        file_path: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Alias for rag_ingest_pdf."""
        return self.rag_ingest_pdf(file_path=file_path, doc_id=doc_id, metadata=metadata)

    def _compute_cosine_sim(self, v1: list[float], v2: list[float]) -> float:
        """Computes cosine similarity between two vector lists."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 <= 1e-9 or norm2 <= 1e-9:
            return 0.0
        return dot / (norm1 * norm2)

    def _compute_bm25_score(self, query_tokens: list[str], doc_text: str) -> float:
        """Computes lexical term frequency score proxy for BM25."""
        if not query_tokens or not doc_text:
            return 0.0
        doc_words = re.findall(r"\w+", doc_text.lower())
        if not doc_words:
            return 0.0
        doc_counter = Counter(doc_words)
        doc_len = len(doc_words)
        avg_doc_len = 50.0

        score = 0.0
        k1 = 1.2
        b = 0.75
        for qt in query_tokens:
            tf = doc_counter.get(qt, 0)
            if tf > 0:
                tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len)))
                score += tf_norm
        return score

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Performs hybrid search combining BM25 lexical ranking and dense vector similarity

        with Reciprocal Rank Fusion (RRF) and parent context resolution.
        """
        # If ES client is connected / mocked, try ES search first
        if self.es is not None:
            try:
                body = {
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"text": {"query": query, "boost": 1.0 - alpha}}},
                            ]
                        }
                    },
                    "size": top_k,
                }
                res = self.es.search(index=self.index_name, body=body)
                hits = res.get("hits", {}).get("hits", [])
                if hits:
                    results = []
                    for h in hits:
                        source = h.get("_source", {})
                        parent_id = source.get("parent_id")
                        parent_text = ""
                        if parent_id:
                            try:
                                p_res = self.es.get(index=self.parent_index_name, id=parent_id)
                                parent_text = p_res.get("_source", {}).get("text", "")
                            except Exception:
                                pass
                            if not parent_text and parent_id in self._in_memory_parents:
                                parent_text = self._in_memory_parents[parent_id].get("text", "")

                        results.append(
                            {
                                "chunk_id": source.get("chunk_id", h.get("_id", "")),
                                "doc_id": source.get("doc_id", ""),
                                "parent_id": parent_id,
                                "text": source.get("text", ""),
                                "parent_text": parent_text or source.get("text", ""),
                                "score": float(h.get("_score", 1.0)),
                                "metadata": source.get("metadata", {}),
                            }
                        )
                    return results
            except Exception:
                pass

        # In-memory hybrid RRF search
        if not self._in_memory_children:
            return []

        query_tokens = [t for t in re.findall(r"\w+", query.lower()) if t not in self.STOP_WORDS]
        query_vec = self._get_embedding(query)

        # 1. Score all child chunks for BM25 and Dense
        scored_items = []
        for child in self._in_memory_children:
            bm25_score = self._compute_bm25_score(query_tokens, child["text"])
            dense_score = self._compute_cosine_sim(query_vec, child.get("vector", []))
            scored_items.append(
                {
                    "child": child,
                    "bm25_score": bm25_score,
                    "dense_score": dense_score,
                }
            )

        # 2. Sort for ranks
        scored_items.sort(key=lambda x: x["bm25_score"], reverse=True)
        for rank, item in enumerate(scored_items, 1):
            item["bm25_rank"] = rank

        scored_items.sort(key=lambda x: x["dense_score"], reverse=True)
        for rank, item in enumerate(scored_items, 1):
            item["dense_rank"] = rank

        # 3. Reciprocal Rank Fusion (RRF)
        k_rrf = 60.0
        for item in scored_items:
            rrf_bm25 = 1.0 / (k_rrf + item["bm25_rank"])
            rrf_dense = 1.0 / (k_rrf + item["dense_rank"])
            item["rrf_score"] = ((1.0 - alpha) * rrf_bm25) + (alpha * rrf_dense)

        scored_items.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_items = scored_items[:top_k]

        # 4. Resolve parent context
        results = []
        for item in top_items:
            c = item["child"]
            parent_id = c.get("parent_id", "")
            parent_text = ""
            if parent_id in self._in_memory_parents:
                parent_text = self._in_memory_parents[parent_id].get("text", "")

            results.append(
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "doc_id": c.get("doc_id", ""),
                    "parent_id": parent_id,
                    "text": c.get("text", ""),
                    "parent_text": parent_text or c.get("text", ""),
                    "score": round(item["rrf_score"] * 100.0, 4),
                    "metadata": c.get("metadata", {}),
                }
            )

        return results

    def rag_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Alias for search."""
        return self.search(query=query, top_k=top_k, alpha=alpha, filter_metadata=filter_metadata)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Alias for search."""
        return self.search(query=query, top_k=top_k, alpha=alpha, filter_metadata=filter_metadata)

    def ask(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> dict[str, Any]:
        """Synthesizes a question answer strictly grounded in retrieved knowledge base chunks

        with structured source citations.
        """
        hits = self.search(query=query, top_k=top_k, alpha=alpha)

        if not hits:
            return {
                "query": query,
                "answer": "No relevant documents found in the knowledge base.",
                "citations": [],
                "sources_count": 0,
                "retrieved_chunks": [],
            }

        citations = []
        context_blocks = []
        for idx, hit in enumerate(hits, 1):
            chunk_text = hit.get("text", "")
            parent_text = hit.get("parent_text", "")
            doc_id = hit.get("doc_id", "unknown")
            chunk_id = hit.get("chunk_id", f"c{idx}")
            parent_id = hit.get("parent_id", "")

            snippet = chunk_text if len(chunk_text) <= 200 else f"{chunk_text[:197]}..."
            vault_doc = self.vault.get_document(doc_id) if hasattr(self, "vault") else None
            citations.append(
                {
                    "citation_id": f"[{idx}]",
                    "doc_id": doc_id,
                    "filename": vault_doc.filename if vault_doc else f"{doc_id}.pdf",
                    "vault_path": vault_doc.vault_path if vault_doc else "",
                    "file_size": vault_doc.file_size_bytes if vault_doc else 0,
                    "mime_type": vault_doc.mime_type if vault_doc else "application/pdf",
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "snippet": snippet,
                    "score": hit.get("score", 0.0),
                }
            )
            context_blocks.append(f"[{idx}] (Doc: {doc_id}): {parent_text or chunk_text}")

        joined_context = "\n\n".join(context_blocks)

        if self.model_client is not None and hasattr(self.model_client, "generate"):
            try:
                prompt = (
                    f"Answer the user query strictly grounded in the provided reference context below.\n"
                    f"Always cite reference numbers (e.g. [1], [2]) directly in the text.\n\n"
                    f"Reference Context:\n{joined_context}\n\n"
                    f"Query: {query}\n\n"
                    f"Grounded Answer:"
                )
                answer_text = self.model_client.generate(
                    prompt=prompt,
                    model=self.config.model_name,
                ).strip()
                if answer_text:
                    return {
                        "query": query,
                        "answer": answer_text,
                        "citations": citations,
                        "sources_count": len(citations),
                        "retrieved_chunks": hits,
                    }
            except Exception as e:
                logger.debug(f"Ollama generation failed: {e}")

        relevant_sentences = []
        for idx, hit in enumerate(hits, 1):
            text = hit.get("text", "")
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            for s in sentences:
                if any(
                    w.lower() in s.lower()
                    for w in re.findall(r"\w+", query)
                    if w.lower() not in self.STOP_WORDS
                ):
                    relevant_sentences.append(f"{s} [{idx}]")
            if not relevant_sentences and sentences:
                relevant_sentences.append(f"{sentences[0]} [{idx}]")

        answer_text = " ".join(relevant_sentences[:4])
        if not answer_text:
            answer_text = f"Based on retrieved documents, relevant information was identified in {hits[0].get('doc_id')} [1]."

        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "sources_count": len(citations),
            "retrieved_chunks": hits,
        }

    def rag_ask(self, query: str, top_k: int = 5, alpha: float = 0.5) -> dict[str, Any]:
        """Alias for ask."""
        return self.ask(query=query, top_k=top_k, alpha=alpha)

    def ask_question(self, query: str, top_k: int = 5, alpha: float = 0.5) -> dict[str, Any]:
        """Alias for ask."""
        return self.ask(query=query, top_k=top_k, alpha=alpha)
