"""Executive Planner & Subagent Orchestrator for WorkOS."""

import inspect
import json
import logging
import re
import uuid
from typing import Any

from config import WorkOSConfig, get_config
from workos_engine.agents.base import BaseSubagent
from workos_engine.agents.doc_agent import DocAgent
from workos_engine.agents.mail_agent import MailAgent
from workos_engine.agents.rag_agent import RAGAgent
from workos_engine.agents.web_agent import WebAgent
from workos_engine.memory.networks import CognitiveMemoryEngine
from workos_engine.types import (
    ExecutionPlan,
    ExecutionResult,
    PlanStep,
    SubagentTask,
)

logger = logging.getLogger(__name__)


def _lookup_exact_variable(
    var_token: str,
    completed_steps: dict[int, PlanStep],
    prev_step: PlanStep | None,
) -> Any:
    """Resolves a single atomic token like '$step_1.saved_path' or '$prev.output'."""
    if not isinstance(var_token, str) or not var_token.startswith("$"):
        return var_token

    token = var_token[1:]  # strip leading $
    parts = token.split(".")
    target_step: PlanStep | None = None

    if parts[0].startswith("step_"):
        try:
            step_id = int(parts[0].replace("step_", ""))
            target_step = completed_steps.get(step_id)
        except ValueError:
            target_step = None
    elif parts[0] in ("prev", "previous"):
        target_step = prev_step

    if not target_step or not target_step.result:
        return var_token

    res = target_step.result
    if len(parts) == 1:
        if res.data is not None:
            return res.data
        if res.artifacts:
            return res.artifacts[0]
        return var_token

    current: Any = res.data
    field_path = parts[1:]

    if field_path and field_path[0] == "data":
        field_path = field_path[1:]
        current = res.data
    elif field_path and field_path[0] == "artifacts":
        return res.artifacts

    for part in field_path:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            # Fallback smart extraction for web URLs
            elif part in ("url", "output_url", "link", "web_url", "source_url"):
                if "url" in current:
                    current = current["url"]
                elif (
                    "results" in current
                    and isinstance(current["results"], list)
                    and current["results"]
                ):
                    first_res = current["results"][0]
                    current = (
                        first_res.get("url") if isinstance(first_res, dict) else str(first_res)
                    )
                elif "pages" in current and isinstance(current["pages"], list) and current["pages"]:
                    first_page = current["pages"][0]
                    current = (
                        first_page.get("url") if isinstance(first_page, dict) else str(first_page)
                    )
                elif (
                    "citations" in current
                    and isinstance(current["citations"], list)
                    and current["citations"]
                ):
                    first_cit = current["citations"][0]
                    current = (
                        first_cit.get("url") if isinstance(first_cit, dict) else str(first_cit)
                    )
                else:
                    return var_token
            # Fallback smart extraction for file paths
            elif part in ("saved_path", "file_path", "path", "attachment", "filename"):
                if res.artifacts:
                    return res.artifacts[0]
                current = (
                    current.get("saved_path")
                    or current.get("file_path")
                    or current.get("path")
                    or var_token
                )
            # Fallback smart extraction for text content
            elif part in ("output", "summary", "content", "text", "body"):
                current = (
                    current.get("content")
                    or current.get("text")
                    or current.get("summary")
                    or current.get("body")
                    or current.get("final_output")
                    or var_token
                )
            else:
                return var_token
        elif hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, list):
            try:
                list_idx = int(part)
                current = current[list_idx]
            except (ValueError, IndexError):
                if (
                    part in ("url", "output_url", "link")
                    and current
                    and isinstance(current[0], dict)
                ):
                    current = current[0].get("url", var_token)
                else:
                    return var_token
        else:
            if part in ("saved_path", "file_path", "artifact") and res.artifacts:
                return res.artifacts[0]
            return var_token

    return current if current is not None else var_token


def _lookup_variable(
    var_ref: str,
    completed_steps: dict[int, PlanStep],
    prev_step: PlanStep | None,
) -> Any:
    """Resolves variable references, supporting exact object lookups and embedded string substitutions."""
    if not isinstance(var_ref, str) or "$" not in var_ref:
        return var_ref

    # 1. Exact match (e.g. "$step_1.data" or "$step_2.saved_path") -> returns raw data object
    exact_pattern = r"^\$(?:step_\d+|prev|previous)(?:\.[a-zA-Z0-9_]+)*$"
    if re.match(exact_pattern, var_ref.strip()):
        return _lookup_exact_variable(var_ref.strip(), completed_steps, prev_step)

    # 2. Embedded string substitution (e.g. "$step_1.creator projects" -> "Guido van Rossum projects")
    def _replace_match(match):
        tok = match.group(0)
        resolved_val = _lookup_exact_variable(tok, completed_steps, prev_step)
        return str(resolved_val) if resolved_val != tok else tok

    token_pattern = r"\$(?:step_\d+|prev|previous)(?:\.[a-zA-Z0-9_]+)*"
    return re.sub(token_pattern, _replace_match, var_ref)


def _resolve_variables_in_dict(
    data: dict[str, Any],
    completed_steps: dict[int, PlanStep],
    prev_step: PlanStep | None,
) -> dict[str, Any]:
    """Recursively resolves $step_X variables inside an input_data dictionary."""
    resolved: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str) and "$" in v:
            resolved[k] = _lookup_variable(v, completed_steps, prev_step)
        elif isinstance(v, dict):
            resolved[k] = _resolve_variables_in_dict(v, completed_steps, prev_step)
        elif isinstance(v, list):
            resolved[k] = [
                _lookup_variable(item, completed_steps, prev_step)
                if isinstance(item, str) and "$" in item
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
        config: WorkOSConfig | None = None,
        memory: CognitiveMemoryEngine | None = None,
        mail_agent: MailAgent | None = None,
        doc_agent: DocAgent | None = None,
        rag_agent: RAGAgent | None = None,
        web_agent: WebAgent | None = None,
        client: Any | None = None,
    ):
        self.config = config or get_config()
        self.memory = memory or CognitiveMemoryEngine(config=self.config)
        self.mail_agent = mail_agent or MailAgent(config=self.config)
        self.doc_agent = doc_agent or DocAgent(config=self.config)
        self.rag_agent = rag_agent or RAGAgent(config=self.config)
        self.web_agent = web_agent or WebAgent(
            config=self.config,
            rag_toolkit=getattr(self.rag_agent, "toolkit", None),
        )

        self.agents: dict[str, BaseSubagent] = {
            "mail_agent": self.mail_agent,
            "doc_agent": self.doc_agent,
            "rag_agent": self.rag_agent,
            "web_agent": self.web_agent,
        }

        self.client = client or self._init_ollama_client()

    def _init_ollama_client(self) -> Any:
        """Initializes the Ollama Client."""
        try:
            from workos_engine.llm_client import OllamaClient

            return OllamaClient(
                base_url=self.config.ollama_base_url,
                default_model=self.config.model_name,
                embedding_model=self.config.embedding_model,
            )
        except Exception as e:
            logger.warning(f"Could not initialize Ollama Client: {e}")
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
                belief_lines = [
                    f"- [{b.wing}/{b.hall}] {b.key}: {b.content}" for b in active_beliefs
                ]
                parts.append("User Beliefs & System Preferences:\n" + "\n".join(belief_lines))
        except Exception as e:
            logger.debug(f"Error getting active beliefs: {e}")

        # 3. Relevant recalled facts and entities
        try:
            recalled = self.memory.recall(query=goal, limit=5)
            if recalled:
                fact_lines = [
                    f"- [{m.network.value}:{m.wing}/{m.hall}] {m.key}: {m.content}"
                    for m in recalled
                ]
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

    def _generate_plan(self, goal: str, context: str | None = None) -> ExecutionPlan:
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
            f"1. Choose assigned_agent from: ['mail_agent', 'doc_agent', 'rag_agent', 'web_agent'].\n"
            f"2. In input_data, specify the 'instruction' (matching one of the agent's tool names or capabilities) and all required tool parameters.\n"
            f"3. You can reference outputs of previous steps using variable interpolation like '$step_1.saved_path' or '$step_1.doc_id'.\n"
            f"4. For live internet searches, external research, or online facts, assign to 'web_agent' with instruction 'web_search' or 'web_research_and_ingest'.\n"
            f"5. Output MUST strictly be valid JSON with keys: 'goal' and 'steps' (array of objects with 'step_id', 'description', 'assigned_agent', 'input_data')."
        )

        if self.client and hasattr(self.client, "generate"):
            try:
                response_text = self.client.generate(
                    prompt=prompt,
                    model=self.config.model_name,
                    format="json",
                    temperature=0.1,
                )
                if response_text:
                    return self._parse_plan_json(response_text, fallback_goal=goal)
            except Exception as e:
                logger.warning(
                    f"Ollama plan generation failed: {e}. Falling back to heuristic planner."
                )

        return self._heuristic_plan_fallback(goal)

    def generate_plan(self, goal: str, context: str | None = None) -> ExecutionPlan:
        """
        Public method to generate a structured execution plan. Delegates to _generate_plan.
        """
        return self._generate_plan(goal, context=context)

    def _parse_plan_json(self, response_text: str, fallback_goal: str) -> ExecutionPlan:
        """Parses LLM JSON response into an ExecutionPlan object with multi-format resilience."""
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            start_obj = text.find("{")
            start_arr = text.find("[")
            if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
                end_obj = text.rfind("}")
                if end_obj > start_obj:
                    text = text[start_obj : end_obj + 1]
            elif start_arr != -1:
                end_arr = text.rfind("]")
                if end_arr > start_arr:
                    text = text[start_arr : end_arr + 1]

        data = json.loads(text)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                pass

        if isinstance(data, list):
            goal = fallback_goal
            raw_steps = data
        elif isinstance(data, dict):
            goal = data.get("goal", fallback_goal)
            raw_steps = data.get("steps", [])
        else:
            return self._heuristic_plan_fallback(fallback_goal)

        if isinstance(raw_steps, dict):
            raw_steps = [raw_steps]
        elif not isinstance(raw_steps, list):
            raw_steps = []

        valid_agents = (
            list(self.agents.keys())
            if hasattr(self, "agents")
            else ["mail_agent", "doc_agent", "rag_agent", "web_agent"]
        )

        steps = []
        for idx, s in enumerate(raw_steps, 1):
            if isinstance(s, str):
                desc = s
                step_id = idx
                assigned_agent = (
                    "web_agent"
                    if any(
                        w in desc.lower() for w in ["web", "online", "search", "google", "internet"]
                    )
                    else "rag_agent"
                )
                input_data = {"instruction": "execute", "query": desc}
            elif isinstance(s, dict):
                try:
                    step_id = int(s.get("step_id", idx))
                except (ValueError, TypeError):
                    step_id = idx
                desc = str(
                    s.get("description") or s.get("desc") or s.get("name") or f"Step {step_id}"
                )
                assigned_agent = (
                    str(s.get("assigned_agent") or s.get("agent") or "").strip().lower()
                )

                # Normalize agent name
                if assigned_agent not in valid_agents:
                    if any(w in assigned_agent for w in ["mail", "email", "inbox"]):
                        assigned_agent = "mail_agent"
                    elif any(w in assigned_agent for w in ["doc", "parse", "pdf", "table"]):
                        assigned_agent = "doc_agent"
                    elif any(
                        w in assigned_agent
                        for w in ["web", "search", "internet", "google", "fetch"]
                    ):
                        assigned_agent = "web_agent"
                    elif any(w in assigned_agent for w in ["rag", "knowledge", "index", "vector"]):
                        assigned_agent = "rag_agent"
                    else:
                        assigned_agent = "web_agent" if "web" in desc.lower() else "rag_agent"

                input_data = s.get("input_data") or s.get("params") or s.get("args") or {}
                if isinstance(input_data, str):
                    input_data = {"instruction": "execute", "query": input_data}
                elif not isinstance(input_data, dict):
                    input_data = {}
            else:
                continue

            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=desc,
                    assigned_agent=assigned_agent,
                    input_data=input_data,
                    status="pending",
                )
            )

        if not steps:
            return self._heuristic_plan_fallback(fallback_goal)

        return ExecutionPlan(goal=goal, steps=steps)

    def _heuristic_plan_fallback(self, goal: str) -> ExecutionPlan:
        """Rule-based heuristic plan generator when LLM is offline or in mock environment."""
        goal_lower = goal.lower()
        steps = []
        step_id = 1

        is_email_search = any(
            w in goal_lower for w in ["email", "mail", "inbox", "sender", "arvind", "find invoice"]
        )
        is_doc_parse = any(
            w in goal_lower
            for w in ["pdf", "invoice", "doc", "document", "parse", "table", "chunk"]
        )
        is_rag_or_search = any(
            w in goal_lower for w in ["rag", "index", "knowledge", "hybrid", "search knowledge"]
        )

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
                    input_data={
                        "instruction": "parse_document",
                        "file_path": "$step_1.saved_path",
                    },
                )
            )
            step_id += 1
            if is_rag_or_search:
                steps.append(
                    PlanStep(
                        step_id=step_id,
                        description="Index parsed content into RAG knowledge base",
                        assigned_agent="rag_agent",
                        input_data={
                            "instruction": "rag_ingest_pdf",
                            "file_path": "$step_1.saved_path",
                        },
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
                        input_data={
                            "instruction": instruction,
                            "subject": goal,
                            "body": goal,
                        },
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
        elif any(
            w in goal_lower
            for w in [
                "search online",
                "search web",
                "web",
                "internet",
                "google",
                "lookup online",
                "latest",
                "news",
                "website",
                "http",
                "url",
            ]
        ):
            steps.append(
                PlanStep(
                    step_id=step_id,
                    description="Search live web and research topic",
                    assigned_agent="web_agent",
                    input_data={"instruction": "web_research_and_ingest", "query": goal},
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
        prev_step: PlanStep | None = None

        for step in plan.steps:
            step.status = "in_progress"

            # 1. Resolve variable references
            resolved_inputs = _resolve_variables_in_dict(
                step.input_data, completed_steps, prev_step
            )
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
            status_str = (
                f"Step {s.step_id} ({s.assigned_agent}): {s.description} -> [{s.status.upper()}]"
            )
            if s.result:
                if s.result.success:
                    status_str += f"\nData: {s.result.data}"
                    if s.result.artifacts:
                        status_str += f"\nArtifacts: {s.result.artifacts}"
                else:
                    status_str += f"\nError: {s.result.error}"
            step_summaries.append(status_str)

        steps_text = "\n\n".join(step_summaries)

        if self.client and hasattr(self.client, "generate"):
            try:
                prompt = (
                    f"You are the WorkOS Executive AI Operating System.\n"
                    f"The user goal was: '{plan.goal}'\n\n"
                    f"Step Execution History & Retrieved Findings:\n{steps_text}\n\n"
                    f"Synthesize a clear, strictly grounded, professional executive brief based on the data above:\n"
                    f"1. ZERO-HALLUCINATION: You MUST base 100% of your facts, tables, entity names, sender addresses, filenames, and numbers exclusively on the Step Execution History above.\n"
                    f"2. If retrieved search results or email lists are empty (`[]`) or no matching records exist, you MUST explicitly state that no matching emails or records were found in the inbox. You are strictly FORBIDDEN from inventing fake filenames (e.g. email1.txt), placeholder accounts, or fake numbers.\n"
                    f"3. When real emails are retrieved, cite real sender addresses, subject lines, dates, and whether they represent replies or outbound messages.\n"
                    f"4. If multiple sources or webpages present differing perspectives, compare the arguments and summarize consensus.\n"
                    f"5. Cite sources using [1], [2] referencing source URLs or documents.\n"
                    f"6. Format mathematical expressions with LaTeX ($...$ or $$...$$) and code with markdown code fences."
                )
                response_text = self.client.generate(
                    prompt=prompt,
                    model=self.config.model_name,
                )
                if response_text:
                    summary = response_text.strip()
                    plan.final_output = summary
                    return summary
            except Exception as e:
                logger.warning(f"Ollama synthesis failed: {e}. Using deterministic synthesis.")

        # Deterministic fallback summary
        all_ok = all(s.status == "completed" for s in plan.steps)
        lines = [
            f"Execution {'completed successfully' if all_ok else 'finished with issues'} for goal: {plan.goal}\n"
        ]
        for s in plan.steps:
            if s.result and s.result.success:
                lines.append(
                    f"- Step {s.step_id} ({s.assigned_agent}): {s.description} -> {s.result.data}"
                )
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
