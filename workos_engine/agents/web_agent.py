"""Web Intelligence Specialist Subagent for WorkOS."""

from __future__ import annotations

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
            "research": "web_research_and_ingest",
            "web_research": "web_research_and_ingest",
        }

        if instruction in mapped_tools:
            tool_name = mapped_tools[instruction]
            try:
                data = self.execute_tool(tool_name, context)
                artifacts = data.get("citations", []) if isinstance(data, dict) else []
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

        # Higher-level web mission -> run micro-ReAct loop
        return super().execute(task)

    execute_task = execute
