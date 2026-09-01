from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class MemoryNetwork(str, Enum):
    FACTS = "facts"
    EXPERIENCES = "experiences"
    ENTITIES = "entities"
    BELIEFS = "beliefs"


class AutonomyLevel(str, Enum):
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
    superseded_by: Optional[str] = None
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
    error: Optional[str] = None
    artifacts: list[str] = field(default_factory=list)


@dataclass
class PlanStep:
    step_id: int
    description: str
    assigned_agent: str
    input_data: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[ExecutionResult] = None


@dataclass
class ExecutionPlan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    final_output: Optional[str] = None
