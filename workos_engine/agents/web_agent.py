"""Web Intelligence Specialist Subagent for WorkOS."""

from __future__ import annotations

import logging
from typing import Any

from config import WorkOSConfig, get_config
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
    ):
        self.config = config or get_config()
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

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """
        Executes a web task dispatched by the ExecutivePlanner.
        """
        instruction = (
            task.instruction.lower().strip()
            if task.instruction
            else task.context.get("instruction", "").lower().strip()
        )
        context = task.context or {}

        try:
            if instruction in ("web_search", "search_web", "search"):
                query = context.get("query", "")
                num_results = int(context.get("num_results", 5))
                results = self.toolkit.search_web(query=query, num_results=num_results)
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data={"results": results, "query": query, "count": len(results)},
                )

            elif instruction in ("web_fetch", "fetch_page", "fetch_page_content", "fetch"):
                url = context.get("url", "")
                data = self.toolkit.fetch_page_content(url=url)
                success = data.get("status") == "success" or bool(data.get("content"))
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=success,
                    data=data,
                )

            elif instruction in (
                "web_research_and_ingest",
                "research_and_ingest",
                "research",
                "web_research",
            ):
                query = context.get("query", "")
                max_pages = int(context.get("max_pages", 3))
                data = self.toolkit.research_and_ingest(query=query, max_pages=max_pages)
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data=data,
                    artifacts=data.get("citations", []),
                )

            # Heuristic default: If query is present, do research; if url is present, fetch
            if "url" in context:
                data = self.toolkit.fetch_page_content(url=context["url"])
                return ExecutionResult(
                    task_id=task.task_id, agent_name=self.name, success=True, data=data
                )
            elif "query" in context:
                data = self.toolkit.research_and_ingest(
                    query=context["query"], max_pages=int(context.get("max_pages", 3))
                )
                return ExecutionResult(
                    task_id=task.task_id, agent_name=self.name, success=True, data=data
                )

            return ExecutionResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error=f"Unrecognized web instruction: '{instruction}'",
            )
        except Exception as e:
            logger.error(f"Error executing WebAgent task {task.task_id}: {e}", exc_info=True)
            return ExecutionResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    execute_task = execute
