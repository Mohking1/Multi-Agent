import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryNetwork(StrEnum):
    FACTS = "facts"
    EXPERIENCES = "experiences"
    ENTITIES = "entities"
    BELIEFS = "beliefs"


class AutonomyLevel(StrEnum):
    FULL = "FULL"
    SUPERVISED = "SUPERVISED"


@dataclass
class ConversationTurn:
    id: str
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.content, str):
            if isinstance(self.content, (dict, list)):
                self.content = json.dumps(self.content)
            else:
                self.content = str(self.content)


@dataclass
class MemoryItem:
    id: str
    network: MemoryNetwork
    wing: str
    hall: str
    key: str
    content: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    superseded_by: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if not isinstance(self.content, str):
            if isinstance(self.content, (dict, list)):
                self.content = json.dumps(self.content)
            else:
                self.content = str(self.content)
        if not isinstance(self.key, str):
            self.key = str(self.key)
        try:
            self.confidence = float(self.confidence)
        except (ValueError, TypeError):
            self.confidence = 1.0


@dataclass
class SubagentTask:
    task_id: str
    agent_name: str
    instruction: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    task_id: str
    agent_name: str
    success: bool
    data: Any = None
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StepReference:
    """Strongly-typed reference to an output field of an earlier plan step."""
    step_id: int
    output_key: str | None = None


@dataclass
class PlanStep:
    step_id: int
    description: str
    assigned_agent: str
    input_data: dict[str, Any] = field(default_factory=dict)
    dependencies: list[int] = field(default_factory=list)
    output_data: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: ExecutionResult | None = None


@dataclass
class ExecutionPlan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    final_output: str | None = None


@dataclass
class BlackboardState:
    """Shared typed state carried across subagent DAG execution steps."""

    root_goal: str
    current_step_index: int = 0
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    step_outputs: dict[int, dict[str, Any]] = field(default_factory=dict)
    shared_data: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_receipt(
        self,
        step_id: int,
        agent_name: str,
        summary: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.completed_steps.append(
            {
                "step_id": step_id,
                "agent": agent_name,
                "summary": summary,
            }
        )
        if data:
            self.step_outputs[step_id] = data
            self.shared_data.update(data)

    def set_step_output(self, step_id: int, output: dict[str, Any]) -> None:
        self.step_outputs[step_id] = output

    def get_step_output(self, step_id: int, key: str | None = None, default: Any = None) -> Any:
        step_out = self.step_outputs.get(step_id, {})
        if key is None:
            return step_out
        return step_out.get(key, default)
