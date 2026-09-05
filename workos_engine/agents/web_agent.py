"""Web Intelligence Specialist Subagent for WorkOS."""

from __future__ import annotations

import json
import logging
from typing import Any

from config import WorkOSConfig
from workos_engine.agents.base import BaseSubagent
from workos_engine.tools.web_tools import WebToolKit
from workos_engine.types import ExecutionResult, SubagentTask

logger = logging.getLogger("workos.agents.web")


class WebAgent(BaseSubagent):
    """
    Web Intelligence Specialist Subagent.
    Enables zero-cloud real-time internet search, deep markdown page extraction,
    and automatic grounding synthesis citations.
    """

    def __init__(
        self,
        config: WorkOSConfig | None = None,
        toolkit: WebToolKit | None = None,
        rag_toolkit: Any | None = None,
        model_client: Any | None = None,
    ):
        super().__init__(config=config, model_client=model_client)
        self.toolkit = toolkit or WebToolKit(config=self.config, rag_toolkit=rag_toolkit)

    @property
    def name(self) -> str:
        return "web_agent"

    @property
    def description(self) -> str:
        return (
            "Specialist agent for live internet search (SearXNG/DuckDuckGo), "
            "deep web article extraction via Trafilatura, and web knowledge grounding."
        )

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Dispatches an individual tool call for the WebAgent."""
        t = tool_name.lower().strip()
        if t in ("web_search", "search_web", "search"):
            query = args.get("query", "")
            num_results = int(args.get("num_results", 5))
            results = self.toolkit.search_web(query=query, num_results=num_results)
            return {"results": results, "query": query, "count": len(results)}
        elif t in ("web_fetch", "fetch_page", "fetch_page_content", "fetch"):
            url = args.get("url", "")
            return self.toolkit.fetch_page_content(url=url)
        elif t in (
            "web_research_and_ingest",
            "research_and_ingest",
            "research",
            "web_research",
        ):
            query = args.get("query", "")
            max_pages = int(args.get("max_pages", 3))
            return self.toolkit.research_and_ingest(query=query, max_pages=max_pages)
        else:
            raise ValueError(f"Unknown web tool: {tool_name}")

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns JSON schema tool definitions for the WebAgent."""
        return [
            {
                "name": "web_search",
                "description": "Searches the live open internet for queries, returning top URLs, titles, and snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string.",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Maximum number of search results (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "web_fetch",
                "description": "Fetches a URL and extracts clean, ad-free Markdown content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The HTTP/HTTPS URL of the webpage to fetch.",
                        }
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "web_research_and_ingest",
                "description": "Performs multi-page web search and deep Markdown extraction, returning grounded citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Topic or research query.",
                        },
                        "max_pages": {
                            "type": "integer",
                            "description": "Maximum pages to fetch and parse (default 3).",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def _generate_search_hypotheses(self, goal: str, context: dict[str, Any]) -> list[str]:
        client = getattr(self, "model_client", None)
        if not client or not hasattr(client, "generate"):
            return [goal]

        prompt = (
            f"You are the WorkOS Web Search Specialist.\n"
            f"User Goal: '{goal}'\n\n"
            f"Formulate 2 to 4 diverse, highly effective web search queries to find authoritative information.\n"
            f"Consider specific technical terms, official portal names, site operators (e.g. site:.gov, site:.org, or official domains if appropriate), or exact phrases.\n\n"
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
            logger.debug(f"Web hypothesis generation fallback: {e}")

        fallback_query = context.get("query") or goal
        return [fallback_query]

    def _retrieve_candidates(
        self, queries: list[str], limit_per_query: int = 5
    ) -> list[dict[str, Any]]:
        all_candidates: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for q in queries:
            try:
                results = self.toolkit.search_web(query=q, num_results=limit_per_query)
                for r in results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_candidates.append(r)
            except Exception as e:
                logger.debug(f"Web search error for query '{q}': {e}")
        return all_candidates

    def _filter_and_rerank_candidates(
        self, goal: str, candidates: list[dict[str, Any]], top_k: int = 3
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not candidates:
            return [], []
        client = getattr(self, "model_client", None)
        if not client or not hasattr(client, "generate"):
            return candidates[:top_k], []

        candidate_snippets = []
        for idx, c in enumerate(candidates[:20], 1):
            snippet = f"[{idx}] Title: {c.get('title')} | URL: {c.get('url')}\nSnippet: {c.get('snippet', '')[:160]}"
            candidate_snippets.append(snippet)

        prompt = (
            f"You are the WorkOS Web Precision Re-Ranker.\n"
            f"User Goal: '{goal}'\n\n"
            f"Candidate search results:\n" + "\n---\n".join(candidate_snippets) + "\n\n"
            f"Task: Evaluate each candidate. REJECT SEO clickbait, social media chatter, irrelevant ads, or spam.\n"
            f"Select the top {top_k} most authoritative, reliable, and directly relevant sources to read deeply.\n\n"
            f"Output strictly a JSON object:\n"
            f"{{\n"
            f'  "selected_indices": [<1-based index numbers of the best {top_k} candidates>],\n'
            f'  "rationale": "<brief explanation of why these were chosen>"\n'
            f"}}\n"
        )
        try:
            res = client.generate(prompt=prompt)
            if res:
                parsed = extract_and_parse_json(res)
                selected_indices = parsed.get("selected_indices", [])
                selected = []
                rejected = []
                for idx, c in enumerate(candidates, 1):
                    if idx in selected_indices:
                        selected.append(c)
                    else:
                        rejected.append(c)
                if selected:
                    return selected[:top_k], rejected
        except Exception as e:
            logger.debug(f"Web re-ranking fallback: {e}")

        return candidates[:top_k], candidates[top_k:]

    def _deep_fetch_and_summarize(
        self, goal: str, selected_candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        citations: list[dict[str, Any]] = []
        extracted_content: list[str] = []

        for idx, c in enumerate(selected_candidates, 1):
            url = c.get("url", "")
            cit_id = f"[{idx}]"
            try:
                page = self.toolkit.fetch_page_content(url=url)
                text = page.get("markdown") or page.get("content") or ""
                title = page.get("title") or c.get("title", f"Source {idx}")
                snippet = text[:400].strip() if text else c.get("snippet", "")
                citations.append(
                    {
                        "citation_id": cit_id,
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": url,
                    }
                )
                if text:
                    extracted_content.append(f"Source {cit_id} ({title} - {url}):\n{text[:1200]}")
            except Exception as e:
                logger.debug(f"Error fetching page {url}: {e}")
                citations.append(
                    {
                        "citation_id": cit_id,
                        "title": c.get("title", f"Source {idx}"),
                        "url": url,
                        "snippet": c.get("snippet", ""),
                        "source": url,
                    }
                )

        combined_text = "\n\n---\n\n".join(extracted_content)
        return {
            "citations": citations,
            "raw_content": combined_text,
            "pages_read_count": len(extracted_content),
        }

    def autonomous_research(self, goal: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Generalized 4-phase autonomous web intelligence pipeline:
        1. Hypothesis Generation: Formulates targeted search queries.
        2. Candidate Retrieval: Fetches search result links across queries.
        3. Precision Re-Ranking: LLM weeds out clickbait/ads, selecting authoritative sources.
        4. Targeted Deep Fetch: Fetches clean markdown and builds grounded citations.
        5. Execution Receipt: Compact summary for the parent blackboard.
        """
        queries = self._generate_search_hypotheses(goal, context)
        candidates = self._retrieve_candidates(queries)
        if not candidates:
            receipt = (
                f"Executed {len(queries)} search queries for '{goal}'. Zero web results found."
            )
            return {
                "status": "completed",
                "query": goal,
                "queries": queries,
                "candidates_count": 0,
                "citations": [],
                "results": [],
                "receipt": receipt,
                "message": receipt,
            }

        selected, rejected = self._filter_and_rerank_candidates(goal, candidates)
        fetch_data = self._deep_fetch_and_summarize(goal, selected)

        receipt = (
            f"Scanned {len(candidates)} candidate URLs across {len(queries)} search queries. "
            f"Precision filtered and deeply extracted {len(selected)} authoritative pages. "
            f"Generated {len(fetch_data['citations'])} verified citations."
        )
        return {
            "status": "completed",
            "query": goal,
            "queries": queries,
            "candidates_count": len(candidates),
            "selected_count": len(selected),
            "citations": fetch_data["citations"],
            "content_summary": fetch_data["raw_content"][:2000],
            "receipt": receipt,
            "message": receipt,
        }

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a web task or runs autonomous micro-ReAct loop."""
        instruction = (
            task.instruction.lower().strip()
            if task.instruction
            else task.context.get("instruction", "").lower().strip()
        )
        context = task.context or {}

        mapped_tools = {
            "web_search": "web_search",
            "search_web": "web_search",
            "search": "web_search",
            "web_fetch": "web_fetch",
            "fetch_page": "web_fetch",
            "fetch_page_content": "web_fetch",
            "fetch": "web_fetch",
            "web_research_and_ingest": "web_research_and_ingest",
            "research_and_ingest": "web_research_and_ingest",
        }

        # Direct mapped tool calls
        if instruction in mapped_tools:
            try:
                data = self.execute_tool(mapped_tools[instruction], context)
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data=data,
                    artifacts=[],
                )
            except Exception as e:
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=False,
                    error=str(e),
                )

        # Autonomous research & web search pipeline
        goal = (
            context.get("goal")
            or context.get("query")
            or context.get("step_description")
            or task.instruction
        )
        try:
            data = self.autonomous_research(goal=goal, context=context)
            artifacts = data.get("citations", []) if isinstance(data, dict) else []
            return ExecutionResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=True,
                data=data,
                artifacts=artifacts,
            )
        except Exception as e:
            logger.exception(f"Error in autonomous_research: {e}")
            return ExecutionResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    execute_task = execute
