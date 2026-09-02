"""Base subagent interface for WorkOS specialist subagents with autonomous micro-ReAct support."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from config import WorkOSConfig, get_config
from workos_engine.types import ExecutionResult, SubagentTask

logger = logging.getLogger(__name__)


class BaseSubagent(ABC):
    """
    Abstract base class for all WorkOS specialist subagents.
    Provides autonomous micro-ReAct loops with narrow tool definitions,
    enabling small local LLMs (3B-7B) to reason and execute multi-turn workflows
    without overwhelming their context window.
    """

    name: str = "base_agent"
    description: str = "Base subagent"

    def __init__(
        self,
        config: WorkOSConfig | None = None,
        model_client: Any | None = None,
    ):
        self.config = config or get_config()
        self.model_client = model_client or self._init_model_client()

    def _init_model_client(self) -> Any:
        """Initializes default Ollama client if available."""
        try:
            from workos_engine.llm_client import OllamaClient

            return OllamaClient(
                base_url=self.config.ollama_base_url,
                default_model=self.config.model_name,
                embedding_model=self.config.embedding_model,
            )
        except Exception:
            return None

    @abstractmethod
    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Executes an individual tool primitive exposed by this subagent."""

    @abstractmethod
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns JSON Schema definitions of the tools exposed by this subagent."""

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """
        Executes a delegated SubagentTask.
        If a direct tool instruction is provided, dispatches directly.
        If a higher-level natural language mission is provided and model client is available,
        runs the autonomous micro-ReAct loop.
        """
        instruction = (task.instruction or "").lower().strip()
        tools = {t["name"].lower(): t for t in self.get_tool_definitions()}

        # 1. Direct tool call dispatch if instruction directly matches a tool name
        if instruction in tools:
            try:
                data = self.execute_tool(instruction, task.context or {})
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data=data,
                )
            except Exception as e:
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=False,
                    error=str(e),
                )

        # 2. Check for explicit unsupported/unknown op strings (single-word invalid instructions)
        if any(w in instruction for w in ["unknown", "unsupported", "invalid", "unrecognized"]) or (
            "_" in instruction and " " not in instruction and instruction not in tools
        ):
            return ExecutionResult(
                task_id=task.task_id,
                agent_name=self.name,
                success=False,
                error=f"Unknown instruction: {task.instruction}",
            )

        # 3. Autonomous micro-ReAct loop for multi-turn reasoning missions
        if self.model_client and hasattr(self.model_client, "generate"):
            return self.run_react_loop(task)

        # 4. Fallback for unrecognized instruction without model client
        return ExecutionResult(
            task_id=task.task_id,
            agent_name=self.name,
            success=False,
            error=f"Unknown instruction: {task.instruction}",
        )

    def run_react_loop(self, task: SubagentTask, max_iterations: int = 3) -> ExecutionResult:
        """
        Executes a sequential Thought -> Action -> Observation micro-ReAct loop.
        Confined strictly to this subagent's narrow toolset.
        """
        mission = (
            task.instruction
            or task.context.get("query")
            or task.context.get("text")
            or "Execute task"
        )
        tool_defs = self.get_tool_definitions()
        tools_summary = json.dumps(
            [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                }
                for t in tool_defs
            ],
            indent=2,
        )

        history: list[str] = []
        artifacts: list[str] = []
        accumulated_data: list[Any] = []

        today_str = datetime.now().strftime("%Y-%m-%d")
        system_prompt = (
            f"You are the {self.name} for the WorkOS operating system.\n"
            f"Role: {self.description}\n"
            f"Current System Date: {today_str}\n\n"
            f"You have access ONLY to the following tools:\n{tools_summary}\n\n"
            f"Your job is to accomplish the assigned mission by calling the available tools sequentially.\n"
            f"Respond ONLY in valid JSON with one of the following formats:\n"
            f"Format 1 (Call a tool):\n"
            f'{{"thought": "<reasoning for next step>", "action": "call_tool", "tool": "<tool_name>", "args": {{<arguments>}}}}\n\n'
            f"Format 2 (Mission complete):\n"
            f'{{"thought": "<final evaluation>", "action": "finish", "findings": "<clear, concise markdown summary of findings>", "data": <optional extracted data or list>}}'
        )

        for iteration in range(1, max_iterations + 1):
            history_str = "\n".join(history) if history else "No previous actions yet."
            prompt = (
                f"Assigned Mission: '{mission}'\n\n"
                f"Context Parameters: {json.dumps(task.context or {})}\n\n"
                f"Action & Observation History:\n{history_str}\n\n"
                f"Iteration {iteration}/{max_iterations}. Decide your next action (respond in JSON):"
            )

            try:
                response = self.model_client.generate(
                    prompt=prompt,
                    system=system_prompt,
                    format="json",
                    temperature=0.1,
                )
                parsed = json.loads(response.strip())
            except Exception as e:
                logger.warning(f"[{self.name}] ReAct step {iteration} JSON parse error: {e}")
                break

            action = parsed.get("action", "")
            thought = parsed.get("thought", "")

            # If model finished mission
            if action == "finish" or "findings" in parsed:
                findings = parsed.get("findings") or thought or "Mission completed."
                data = parsed.get("data") if parsed.get("data") is not None else accumulated_data
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data={"findings": findings, "details": data},
                    artifacts=artifacts,
                )

            # If model called a tool
            if action == "call_tool" or "tool" in parsed:
                tool_name = parsed.get("tool", "")
                args = parsed.get("args") or {}

                history.append(f"Step {iteration} Action: {tool_name}({json.dumps(args)})")
                try:
                    obs = self.execute_tool(tool_name, args)
                    accumulated_data.append(obs)

                    if (
                        isinstance(obs, list)
                        and obs
                        and isinstance(obs[0], dict)
                        and "uid" in obs[0]
                    ):
                        formatted_items = [
                            f"UID {m.get('uid')}: From <{m.get('from')}> | Subj: {m.get('subject')} | Date: {m.get('date')}"
                            for m in obs[:25]
                        ]
                        obs_snippet = f"Retrieved {len(obs)} emails:\n" + "\n".join(formatted_items)
                    elif (
                        isinstance(obs, list)
                        and obs
                        and isinstance(obs[0], dict)
                        and "name" in obs[0]
                        and "delimiter" in obs[0]
                    ):
                        formatted_folders = [
                            f"- {f.get('name')} (delim: '{f.get('delimiter')}')" for f in obs
                        ]
                        obs_snippet = f"Folders ({len(obs)}):\n" + "\n".join(formatted_folders[:30])
                    else:
                        obs_snippet = str(obs)[:1200]

                    history.append(f"Step {iteration} Observation: {obs_snippet}")

                    if isinstance(obs, dict) and "saved_path" in obs:
                        artifacts.append(obs["saved_path"])
                except Exception as err:
                    history.append(f"Step {iteration} Observation Error: {err}")
            else:
                history.append(f"Step {iteration}: No valid action recognized.")

        # Final consolidation of findings from gathered tool observations
        findings = ""
        if accumulated_data and self.model_client and hasattr(self.model_client, "generate"):
            try:
                obs_summary_prompt = (
                    f"Mission: '{mission}'\n\n"
                    f"Observed Results from Tool Calls:\n"
                    + "\n".join(history)
                    + "\n\nSynthesize your final findings based strictly on the observations above."
                )
                summary_res = self.model_client.generate(
                    prompt=obs_summary_prompt,
                    system=f"You are the {self.name}. Summarize your findings based ONLY on your tool observations.",
                    temperature=0.1,
                )
                if summary_res:
                    findings = summary_res.strip()
            except Exception as e:
                logger.debug(f"[{self.name}] Error consolidating findings: {e}")

        if not findings:
            findings = f"Completed {len(history)} execution steps for mission: {mission}"

        return ExecutionResult(
            task_id=task.task_id,
            agent_name=self.name,
            success=True,
            data={"findings": findings, "steps": history, "details": accumulated_data},
            artifacts=artifacts,
        )
