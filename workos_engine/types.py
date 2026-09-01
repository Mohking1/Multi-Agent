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


@dataclass
class PlanStep:
    step_id: int
    description: str
    assigned_agent: str
    input_data: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: ExecutionResult | None = None


@dataclass
class ExecutionPlan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    final_output: str | None = None
