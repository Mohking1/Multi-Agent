import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("multi_agent.mcp")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MAILDOC_SERVER_SCRIPT = os.getenv(
    "MAILDOC_SERVER_SCRIPT", os.path.join(PARENT_DIR, "maildoc", "server.py")
)
RAG_SERVER_SCRIPT = os.getenv(
    "RAG_SERVER_SCRIPT", os.path.join(PARENT_DIR, "RAG", "mcp_server.py")
)


class MCPClientManager:
    """Manages connections and tool execution across local MCP stdio servers."""

    def __init__(self):
        self.server_configs: dict[str, StdioServerParameters] = {}
        self._setup_server_configs()

    def _setup_server_configs(self):
        env = dict(os.environ)

        if os.path.exists(MAILDOC_SERVER_SCRIPT):
            self.server_configs["maildoc"] = StdioServerParameters(
                command=sys.executable,
                args=[MAILDOC_SERVER_SCRIPT],
                env={**env, "PYTHONPATH": os.path.dirname(MAILDOC_SERVER_SCRIPT)},
            )

        if os.path.exists(RAG_SERVER_SCRIPT):
            self.server_configs["rag"] = StdioServerParameters(
                command=sys.executable,
                args=[RAG_SERVER_SCRIPT],
                env={**env, "PYTHONPATH": os.path.dirname(RAG_SERVER_SCRIPT)},
            )

    @asynccontextmanager
    async def get_session(self, server_name: str):
        """Open a stdio client session with the target MCP server."""
        if server_name not in self.server_configs:
            raise ValueError(
                f"Unknown MCP server '{server_name}'. Available: {list(self.server_configs.keys())}"
            )

        params = self.server_configs[server_name]
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            yield session

    async def list_server_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List all tools exposed by an MCP server."""
        try:
            async with self.get_session(server_name) as session:
                tools_result = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in tools_result.tools
                ]
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Failed to list tools from MCP server '%s': %s", server_name, e
            )
            return []

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Call a specific tool on an MCP server over stdio."""
        args = arguments or {}
        try:
            async with self.get_session(server_name) as session:
                result = await session.call_tool(tool_name, arguments=args)
                if result.isError:
                    error_text = " ".join(
                        [c.text for c in result.content if hasattr(c, "text")]
                    )
                    return {"status": "error", "error": error_text}

                content_parts = []
                for part in result.content:
                    if hasattr(part, "text"):
                        content_parts.append(part.text)
                return "\n".join(content_parts) if content_parts else result
        except Exception as e:  # noqa: BLE001
            logger.error(
                "MCP tool execution error (%s/%s): %s", server_name, tool_name, e
            )
            return {"status": "error", "error": str(e)}

    def call_tool_sync(
        self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Synchronous wrapper to call an MCP tool."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In an existing loop, run in a separate thread/task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self.call_tool(server_name, tool_name, arguments)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.call_tool(server_name, tool_name, arguments)
                )
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)}


mcp_manager = MCPClientManager()
