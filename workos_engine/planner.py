"""Executive Planner & Subagent Orchestrator for WorkOS."""
import asyncio
import inspect
import json
import logging
import re
import uuid
from typing import Any, Optional

from config import WorkOSConfig, get_config
from workos_engine.agents.base import BaseSubagent
from workos_engine.agents.doc_agent import DocAgent
from workos_engine.agents.mail_agent import MailAgent
from workos_engine.agents.rag_agent import RAGAgent
from workos_engine.memory.networks import CognitiveMemoryEngine
from workos_engine.types import (
    ExecutionPlan,
    ExecutionResult,
    MemoryNetwork,
    PlanStep,
    SubagentTask,
)

logger = logging.getLogger(__name__)


def _lookup_variable(
    var_ref: str,
    completed_steps: dict[int, PlanStep],
    prev_step: Optional[PlanStep],
) -> Any:
    """Resolves a variable reference string (e.g. '$step_1.saved_path' or '$prev.total')."""
    if not isinstance(var_ref, str) or not var_ref.startswith("$"):
        return var_ref

    token = var_ref[1:]  # strip leading $
    parts = token.split(".")
    target_step: Optional[PlanStep] = None

    if parts[0].startswith("step_"):
        try:
            step_id = int(parts[0].replace("step_", ""))
            target_step = completed_steps.get(step_id)
        except ValueError:
            target_step = None
    elif parts[0] in ("prev", "previous"):
        target_step = prev_step

    if not target_step or not target_step.result:
        return var_ref

    res = target_step.result
    if len(parts) == 1:
        if res.data is not None:
            return res.data
        if res.artifacts:
            return res.artifacts[0]
        return var_ref

    # Navigate properties
    current: Any = res.data
    field_path = parts[1:]

    if field_path[0] == "data":
        field_path = field_path[1:]
        current = res.data
    elif field_path[0] == "artifacts":
        return res.artifacts
    elif field_path[0] in ("saved_path", "file_path") and (
        current is None or (isinstance(current, dict) and field_path[0] not in current)
    ):
        if res.artifacts:
            return res.artifacts[0]

    for part in field_path:
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            if part in ("saved_path", "file_path", "artifact") and res.artifacts:
                return res.artifacts[0]
            return var_ref

    return current if current is not None else var_ref


def _resolve_variables_in_dict(
    data: dict[str, Any],
    completed_steps: dict[int, PlanStep],
    prev_step: Optional[PlanStep],
) -> dict[str, Any]:
    """Recursively resolves $step_X variables inside an input_data dictionary."""
    resolved: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str) and v.startswith("$"):
            resolved[k] = _lookup_variable(v, completed_steps, prev_step)
        elif isinstance(v, dict):
            resolved[k] = _resolve_variables_in_dict(v, completed_steps, prev_step)
        elif isinstance(v, list):
            resolved[k] = [
                _lookup_variable(item, completed_steps, prev_step)
                if isinstance(item, str) and item.startswith("$")
                else item
                for item in v
            ]
        else:
            resolved[k] = v
    return resolved


class ExecutivePlanner:
    """
    Central Executive Planner for WorkOS.
    Injects cognitive memory context, dynamically constructs structured execution plans with Gemini,
    dispatches steps to MailAgent, DocAgent, and RAGAgent, synthesizes final outputs,
    and triggers background memory reflection.
    """

    def __init__(
        self,
        config: Optional[WorkOSConfig] = None,
        memory: Optional[CognitiveMemoryEngine] = None,
        mail_agent: Optional[MailAgent] = None,
        doc_agent: Optional[DocAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        client: Optional[Any] = None,
    ):
        self.config = config or get_config()
        self.memory = memory or CognitiveMemoryEngine(config=self.config)
        self.mail_agent = mail_agent or MailAgent(config=self.config)
        self.doc_agent = doc_agent or DocAgent(config=self.config)
        self.rag_agent = rag_agent or RAGAgent(config=self.config)

        self.agents: dict[str, BaseSubagent] = {
            "mail_agent": self.mail_agent,
            "doc_agent": self.doc_agent,
            "rag_agent": self.rag_agent,
        }

        self.client = client or self._init_genai_client()

    def _init_genai_client(self) -> Any:
        """Initializes the Google GenAI Client if available."""
        try:
            from google import genai

            api_key = self.config.gemini_api_key or ""
            return genai.Client(api_key=api_key)
        except Exception as e:
            logger.warning(f"Could not initialize Google GenAI Client: {e}")
            return None

    def build_planning_context(self, goal: str) -> str:
        """
        Gathers active beliefs, spatial Loci memory summary, and relevant recalled facts
        to build a compact context prompt for planning.
        """
        parts = []

        # 1. Spatial index summary
        try:
            summary = self.memory.get_context_index_summary()
            if summary:
                parts.append(f"Spatial Memory Loci Index:\n{summary}")
        except Exception as e:
            logger.debug(f"Error getting spatial summary: {e}")

        # 2. Active beliefs and preferences
        try:
            active_beliefs = self.memory.get_active_beliefs()
            if active_beliefs:
                belief_lines = [f"- [{b.wing}/{b.hall}] {b.key}: {b.content}" for b in active_beliefs]
                parts.append("User Beliefs & System Preferences:\n" + "\n".join(belief_lines))
        except Exception as e:
            logger.debug(f"Error getting active beliefs: {e}")

        # 3. Relevant recalled facts and entities
        try:
            recalled = self.memory.recall(query=goal, limit=5)
            if recalled:
                fact_lines = [f"- [{m.network.value}:{m.wing}/{m.hall}] {m.key}: {m.content}" for m in recalled]
                parts.append("Relevant Recalled Knowledge:\n" + "\n".join(fact_lines))
        except Exception as e:
            logger.debug(f"Error recalling facts for goal: {e}")

        return "\n\n".join(parts)

    def _collect_subagent_tool_registry(self) -> dict[str, list[dict[str, Any]]]:
        """Collects all tool definitions from each registered specialist agent."""
        registry = {}
        for agent_name, agent in self.agents.items():
            try:
                registry[agent_name] = agent.get_tool_definitions()
            except Exception as e:
                logger.warning(f"Error getting tool definitions for {agent_name}: {e}")
                registry[agent_name] = []
        return registry

    def _generate_plan(self, goal: str, context: Optional[str] = None) -> ExecutionPlan:
        """
        Internal implementation of dynamic multi-step ExecutionPlan construction.
        """
        ctx_text = context if context is not None else self.build_planning_context(goal)
        tool_registry = self._collect_subagent_tool_registry()

        tools_json = json.dumps(tool_registry, indent=2)

        prompt = (
            f"You are the WorkOS Executive AI Planner. Formulate an optimal, step-by-step execution plan to accomplish the user goal.\n\n"
            f"Goal: {goal}\n\n"
            f"Available Subagent Tools Registry:\n{tools_json}\n\n"
            f"Cognitive Context & Memory:\n{ctx_text}\n\n"
            f"Rules:\n"
            f"1. Choose assigned_agent from: ['mail_agent', 'doc_agent', 'rag_agent'].\n"
            f"2. In input_data, specify the 'instruction' (matching one of the agent's tool names or capabilities) and all required tool parameters.\n"
            f"3. You can reference outputs of previous steps using variable interpolation like '$step_1.saved_path' or '$step_1.doc_id'.\n"
            f"4. Output MUST strictly be valid JSON with keys: 'goal' and 'steps' (array of objects with 'step_id', 'description', 'assigned_agent', 'input_data')."
        )

        if self.client and hasattr(self.client, "models"):
            try:
                from google.genai import types

                gen_config = types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=gen_config,
                )
                if response and response.text:
                    return self._parse_plan_json(response.text, fallback_goal=goal)
            except Exception as e:
                logger.warning(f"LLM plan generation failed: {e}. Falling back to heuristic planner.")

        return self._heuristic_plan_fallback(goal)

    def generate_plan(self, goal: str, context: Optional[str] = None) -> ExecutionPlan:
        """
        Public method to generate a structured execution plan. Delegates to _generate_plan.
        """
        return self._generate_plan(goal, context=context)

    def _parse_plan_json(self, response_text: str, fallback_goal: str) -> ExecutionPlan:
        """Parses LLM JSON response into an ExecutionPlan object."""
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        data = json.loads(text)
        goal = data.get("goal", fallback_goal)
        raw_steps = data.get("steps", [])
        steps = []
        for idx, s in enumerate(raw_steps, 1):
            step_id = s.get("step_id", idx)
            desc = s.get("description", f"Step {step_id}")
            assigned_agent = s.get("assigned_agent", "rag_agent")
            input_data = s.get("input_data", {})
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=desc,
                    assigned_agent=assigned_agent,
                    input_data=input_data,
                    status="pending",
                )
            )
        return ExecutionPlan(goal=goal, steps=steps)

    def _heuristic_plan_fallback(self, goal: str) -> ExecutionPlan:
        """Rule-based heuristic plan generator when LLM is offline or in mock environment."""
        goal_lower = goal.lower()
        steps = []
        step_id = 1

        is_email_search = any(w in goal_lower for w in ["email", "mail", "inbox", "sender", "arvind", "find invoice"])
        is_doc_parse = any(w in goal_lower for w in ["pdf", "invoice", "doc", "document", "parse", "table", "chunk"])
        is_rag_or_search = any(w in goal_lower for w in ["rag", "index", "knowledge", "hybrid", "search knowledge"])

        if is_email_search and is_doc_parse:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description="Search and download attachment from email",
                    assigned_agent="mail_agent",
                    input_data={"instruction": "download_attachment", "query": goal},
                )
            )
            step_id += 1
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description="Parse downloaded document",
                    assigned_agent="doc_agent",
                    input_data={"instruction": "parse_document", "file_path": "$step_1.saved_path"},
                )
            )
            step_id += 1
            if is_rag_or_search:
                steps.append(
                    PlanStep(
                        step_id=step_id,
                        description="Index parsed content into RAG knowledge base",
                        assigned_agent="rag_agent",
                        input_data={"instruction": "rag_ingest_pdf", "file_path": "$step_1.saved_path"},
                    )
                )
        elif is_email_search:
            if "send" in goal_lower or "draft" in goal_lower:
                instruction = "send_email" if "send" in goal_lower else "create_draft"
                steps.append(
                    PlanStep(
                        step_id=step_id,
                        description=f"Execute email {instruction}",
                        assigned_agent="mail_agent",
                        input_data={"instruction": instruction, "subject": goal, "body": goal},
                    )
                )
            else:
                steps.append(
                    PlanStep(
                        step_id=step_id,
                        description="Search emails",
                        assigned_agent="mail_agent",
                        input_data={"instruction": "search_emails", "text": goal},
                    )
                )
        elif is_doc_parse:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description="Parse document",
                    assigned_agent="doc_agent",
                    input_data={"instruction": "parse_document", "file_path": goal},
                )
            )
        else:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description="Query RAG knowledge base",
                    assigned_agent="rag_agent",
                    input_data={"instruction": "rag_ask", "query": goal},
                )
            )

        return ExecutionPlan(goal=goal, steps=steps)

    async def execute_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Executes each PlanStep in the ExecutionPlan sequentially, resolving dynamic variables,
        tracking status transitions (pending -> in_progress -> completed / failed),
        and updating step results.
        """
        completed_steps: dict[int, PlanStep] = {}
        prev_step: Optional[PlanStep] = None

        for step in plan.steps:
            step.status = "in_progress"

            # 1. Resolve variable references
            resolved_inputs = _resolve_variables_in_dict(step.input_data, completed_steps, prev_step)
            step.input_data = resolved_inputs

            # 2. Check assigned agent
            agent = self.agents.get(step.assigned_agent)
            if not agent:
                step.status = "failed"
                step.result = ExecutionResult(
                    task_id=str(step.step_id),
                    agent_name=step.assigned_agent,
                    success=False,
                    error=f"Unknown agent '{step.assigned_agent}' in plan step {step.step_id}",
                )
                prev_step = step
                continue

            # 3. Extract instruction and context
            instruction = resolved_inputs.get("instruction") or step.description
            context_args = {k: v for k, v in resolved_inputs.items() if k != "instruction"}

            task = SubagentTask(
                task_id=f"step_{step.step_id}_{uuid.uuid4().hex[:6]}",
                agent_name=step.assigned_agent,
                instruction=instruction,
                context=context_args,
            )

            # 4. Dispatch to subagent
            try:
                if inspect.iscoroutinefunction(agent.execute):
                    res = await agent.execute(task)
                else:
                    res = agent.execute(task)
                    if inspect.iscoroutine(res):
                        res = await res
                step.result = res
                step.status = "completed" if res.success else "failed"
            except Exception as e:
                logger.exception(f"Error executing step {step.step_id}: {e}")
                step.status = "failed"
                step.result = ExecutionResult(
                    task_id=task.task_id,
                    agent_name=step.assigned_agent,
                    success=False,
                    error=str(e),
                )

            completed_steps[step.step_id] = step
            prev_step = step

        return plan

    def synthesize_response(self, plan: ExecutionPlan) -> str:
        """
        Synthesizes a grounded final user-facing response from step execution results.
        """
        step_summaries = []
        for s in plan.steps:
            status_str = f"Step {s.step_id} ({s.assigned_agent}): {s.description} -> [{s.status.upper()}]"
            if s.result:
                if s.result.success:
                    status_str += f"\nData: {s.result.data}"
                    if s.result.artifacts:
                        status_str += f"\nArtifacts: {s.result.artifacts}"
                else:
                    status_str += f"\nError: {s.result.error}"
            step_summaries.append(status_str)

        steps_text = "\n\n".join(step_summaries)

        if self.client and hasattr(self.client, "models") and self.config.gemini_api_key:
            try:
                prompt = (
                    f"You are the WorkOS Executive AI Operating System.\n"
                    f"The user goal was: '{plan.goal}'\n\n"
                    f"Step Execution History:\n{steps_text}\n\n"
                    f"Synthesize a clear, direct, professional response grounded strictly in the data above. "
                    f"Include key metrics, extracted entities, filenames, or failure reasons if applicable."
                )
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                )
                if response and response.text:
                    summary = response.text.strip()
                    plan.final_output = summary
                    return summary
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}. Using deterministic synthesis.")

        # Deterministic fallback summary
        all_ok = all(s.status == "completed" for s in plan.steps)
        lines = [f"Execution {'completed successfully' if all_ok else 'finished with issues'} for goal: {plan.goal}\n"]
        for s in plan.steps:
            if s.result and s.result.success:
                lines.append(f"- Step {s.step_id} ({s.assigned_agent}): {s.description} -> {s.result.data}")
            elif s.result:
                lines.append(f"- Step {s.step_id} ({s.assigned_agent}) FAILED: {s.result.error}")
            else:
                lines.append(f"- Step {s.step_id} ({s.assigned_agent}) [{s.status}]")

        summary = "\n".join(lines)
        plan.final_output = summary
        return summary

    # Alias for compatibility with internal callers
    synthesize_output = synthesize_response

    async def reflect_async(self, goal: str, plan: ExecutionPlan) -> None:
        """
        Asynchronous memory reflection hook that retains the execution experience,
        learned facts, and beliefs in CognitiveMemoryEngine.
        """
        try:
            is_success = all(s.status == "completed" for s in plan.steps)
            status_label = "completed" if is_success else "failed"
            summary = (
                f"Goal: {goal} | Status: {status_label} | "
                f"Steps: {len(plan.steps)} | Output: {plan.final_output or 'None'}"
            )
            key = f"goal_{abs(hash(goal)) % 1000000}"

            # 1. Retain experience directly
            self.memory.retain_experience(
                wing="workflows",
                hall="executions",
                key=key,
                content=summary,
                metadata={
                    "goal": goal,
                    "success": is_success,
                    "steps_count": len(plan.steps),
                },
            )

            # 2. Invoke reflector logic
            events = [
                {"type": "user_goal", "content": goal},
                {
                    "type": "plan_steps",
                    "steps": [
                        {
                            "step_id": s.step_id,
                            "description": s.description,
                            "agent": s.assigned_agent,
                            "status": s.status,
                            "data": s.result.data if s.result else None,
                        }
                        for s in plan.steps
                    ],
                },
                {"type": "final_output", "content": plan.final_output or ""},
            ]
            self.memory.reflect_and_update(
                conversation_events=events,
                model_client=self.client,
            )
        except Exception as e:
            logger.warning(f"Async memory reflection error: {e}")

    # Alias for compatibility with internal callers
    async_reflect = reflect_async

    async def run_goal(self, goal: str, user_id: str = "default_user") -> ExecutionPlan:
        """
        Receives user goal, injects cognitive memory context, dynamically constructs structured execution plans,
        dispatches steps to MailAgent, DocAgent, and RAGAgent, synthesizes final outputs,
        and triggers background memory reflection.
        """
        # 1. Inject cognitive memory context
        context = self.build_planning_context(goal)

        # 2. Generate structured execution plan
        plan = self.generate_plan(goal, context=context)

        # 3. Execute plan steps across specialist agents
        executed_plan = await self.execute_plan(plan)

        # 4. Synthesize final grounded response
        self.synthesize_response(executed_plan)

        # 5. Trigger async reflection
        await self.reflect_async(goal, executed_plan)

        return executed_plan

    # Alias for compatibility with internal callers
    plan_and_execute = run_goal
