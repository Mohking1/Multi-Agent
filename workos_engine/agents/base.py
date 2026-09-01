"""Base subagent interface for WorkOS specialist subagents."""

from abc import ABC, abstractmethod
from typing import Any

from config import WorkOSConfig, get_config
from workos_engine.types import ExecutionResult, SubagentTask


class BaseSubagent(ABC):
    """Abstract base class for all WorkOS specialist subagents."""

    name: str = "base_agent"
    description: str = "Base subagent"

    def __init__(self, config: WorkOSConfig | None = None):
        self.config = config or get_config()

    @abstractmethod
    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a delegated SubagentTask and returns an ExecutionResult."""

    @abstractmethod
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns JSON Schema definitions of the tools exposed by this subagent for LLM planning."""
