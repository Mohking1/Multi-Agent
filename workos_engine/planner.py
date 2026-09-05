"""Executive Planner & Subagent Orchestrator for WorkOS."""

import inspect
import json
import logging
import re
import uuid
from datetime import datetime
from graphlib import CycleError, TopologicalSorter
from typing import Any

from config import WorkOSConfig, get_config
from workos_engine.agents.base import BaseSubagent
from workos_engine.agents.doc_agent import DocAgent
from workos_engine.agents.mail_agent import MailAgent
from workos_engine.agents.rag_agent import RAGAgent
from workos_engine.agents.web_agent import WebAgent
from workos_engine.debug import get_tracer
from workos_engine.llm_client import extract_and_parse_json
from workos_engine.memory.networks import CognitiveMemoryEngine
from workos_engine.types import (
    BlackboardState,
    ExecutionPlan,
    ExecutionResult,
    MemoryNetwork,
    PlanStep,
    StepReference,
    SubagentTask,
)

logger = logging.getLogger(__name__)


class PlanParseError(RuntimeError):
    """Raised when an LLM response cannot be parsed into a valid execution plan."""


def _lookup_exact_variable(
    var_token: Any,
    completed_steps: dict[int, PlanStep],
    prev_step: PlanStep | None,
    blackboard: BlackboardState | None = None,
) -> Any:
    """Resolves a typed StepReference or atomic string token like '$step_1.saved_path' or '$prev.data'."""
    if isinstance(var_token, StepReference):
        if blackboard:
            return blackboard.get_step_output(var_token.step_id, var_token.output_key)
        target = completed_steps.get(var_token.step_id)
        if target and target.result:
            if var_token.output_key and isinstance(target.result.data, dict):
                return target.result.data.get(var_token.output_key)
            return target.result.data
        return None

    if not isinstance(var_token, str) or not var_token.startswith("$"):
        return var_token

    token = var_token[1:].strip()  # strip leading $
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
    field_path = parts[1:]
    if field_path[0] == "artifacts":
        if len(field_path) > 1:
            try:
                idx = int(field_path[1])
                return res.artifacts[idx] if len(res.artifacts) > idx else var_token
            except ValueError:
                pass
        return res.artifacts if res.artifacts else var_token
    if field_path[0] == "artifact":
        return res.artifacts[0] if res.artifacts else var_token
    if field_path[0] == "saved_paths":
        if isinstance(res.data, dict) and "saved_paths" in res.data:
            return res.data["saved_paths"]
        return res.artifacts if res.artifacts else var_token
    if field_path[0] == "saved_path":
        if isinstance(res.data, dict) and "saved_path" in res.data:
            return res.data["saved_path"]
        return res.artifacts[0] if res.artifacts else var_token
    if field_path[0] == "data":
        field_path = field_path[1:]
        if not field_path:
            return res.data if res.data is not None else var_token

    current: Any = res.data
    for part in field_path:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            elif part in ("saved_path", "file_path", "path") and res.artifacts:
                return res.artifacts[0]
            else:
                return var_token
        elif hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return var_token
        else:
            if part in ("saved_path", "file_path") and res.artifacts:
                return res.artifacts[0]
            return var_token

    return current if current is not None else var_token


def _lookup_variable(
    var_ref: Any,
    completed_steps: dict[int, PlanStep],
    prev_step: PlanStep | None,
    blackboard: BlackboardState | None = None,
) -> Any:
    """Resolves variable references, supporting StepReference, exact lookups, and embedded string substitutions."""
    if isinstance(var_ref, StepReference):
        return _lookup_exact_variable(var_ref, completed_steps, prev_step, blackboard)

    if not isinstance(var_ref, str) or "$" not in var_ref:
        return var_ref

    # 1. Exact match (e.g. "$step_1.data" or "$step_2.saved_path")
    exact_pattern = r"^\$(?:step_\d+|prev|previous)(?:\.[a-zA-Z0-9_]+)*$"
    if re.match(exact_pattern, var_ref.strip()):
        return _lookup_exact_variable(var_ref.strip(), completed_steps, prev_step, blackboard)

    # 2. Embedded string substitution (e.g. "$step_1.creator projects")
    def _replace_match(match):
        tok = match.group(0)
        resolved_val = _lookup_exact_variable(tok, completed_steps, prev_step, blackboard)
        return str(resolved_val) if resolved_val != tok else tok

    token_pattern = r"\$(?:step_\d+|prev|previous)(?:\.[a-zA-Z0-9_]+)*"
    return re.sub(token_pattern, _replace_match, var_ref)


def _resolve_variables_in_dict(
    data: dict[str, Any],
    completed_steps: dict[int, PlanStep],
    prev_step: PlanStep | None,
    blackboard: BlackboardState | None = None,
) -> dict[str, Any]:
    """Recursively resolves $step_X variables inside an input_data dictionary."""
    resolved: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, StepReference):
            resolved[k] = _lookup_exact_variable(v, completed_steps, prev_step, blackboard)
        elif isinstance(v, str) and "$" in v:
            resolved[k] = _lookup_variable(v, completed_steps, prev_step, blackboard)
        elif isinstance(v, dict):
            resolved[k] = _resolve_variables_in_dict(v, completed_steps, prev_step, blackboard)
        elif isinstance(v, list):
            resolved[k] = [
                _lookup_variable(item, completed_steps, prev_step, blackboard)
                if (isinstance(item, (str, StepReference)) and (isinstance(item, StepReference) or "$" in item))
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
        """Initializes the configured LLM Client (Gemini or Ollama)."""
        try:
            from workos_engine.llm_client import get_llm_client

            return get_llm_client(self.config)
        except Exception as e:
            logger.warning(f"Could not initialize LLM Client: {e}")
            return None

    def build_planning_context(self, goal: str, session_id: str | None = None) -> str:
        """
        Gathers active beliefs, session conversation history, and relevant recalled
        domain facts/entities/experiences to build a cohesive context prompt for planning.
        """
        parts = [f"Current System Date: {datetime.now().strftime('%Y-%m-%d')}"]

        # 1. Multi-turn dialogue history from active session drawer (MemPalace pattern)
        if session_id:
            try:
                turns = self.memory.get_session_dialogue(session_id=session_id, limit=8)
                if turns:
                    dialogue_lines = [f"  [{t.role.upper()}]: {t.content}" for t in turns]
                    parts.append(
                        "Recent Conversation History (Active Session):\n"
                        + "\n".join(dialogue_lines)
                    )
            except Exception as e:
                logger.debug(f"Error getting session dialogue: {e}")

        # 2. Spatial index summary
        try:
            summary = self.memory.get_context_index_summary()
            if summary:
                parts.append(f"Spatial Memory Loci Index:\n{summary}")
        except Exception as e:
            logger.debug(f"Error getting spatial summary: {e}")

        # 3. Active beliefs and preferences (Hindsight mental models)
        try:
            active_beliefs = self.memory.get_active_beliefs()
            if active_beliefs:
                belief_lines = [
                    f"- [{b.wing}/{b.hall}] {b.key}: {b.content[:150]}" for b in active_beliefs
                ]
                parts.append("User Beliefs & System Preferences:\n" + "\n".join(belief_lines))
        except Exception as e:
            logger.debug(f"Error getting active beliefs: {e}")

        # 4. Relevant recalled domain facts, entities, and past experiences (Hindsight Recall)
        try:
            recalled_facts = self.memory.recall(
                query=goal, network=MemoryNetwork.FACTS, limit=8
            )
            recalled_entities = self.memory.recall(
                query=goal, network=MemoryNetwork.ENTITIES, limit=4
            )
            recalled_experiences = self.memory.recall(
                query=goal, network=MemoryNetwork.EXPERIENCES, limit=4
            )

            clean_facts = []
            for m in recalled_facts + recalled_entities + recalled_experiences:
                clean_facts.append(f"- [{m.network.value}:{m.wing}:{m.key}] {m.content}")
            if clean_facts:
                parts.append(
                    "Relevant Recalled Knowledge & Verified Facts:\n" + "\n".join(clean_facts)
                )
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
        """Internal implementation of dynamic multi-step ExecutionPlan construction."""
        ctx_text = context if context is not None else self.build_planning_context(goal)

        system_prompt = (
            "You are the WorkOS Executive Multi-Agent Planner.\n"
            "You NEVER execute or answer the user request directly.\n"
            'Your ONLY role is to output a JSON object containing the "goal" and a sequential "steps" array decomposing the request across specialist agents:\n\n'
            "Specialist Agents:\n"
            "- 'mail_agent': Email management, search, folder creation, moving, and organizing in the inbox.\n"
            "- 'web_agent': Live web search, news, portals, verifying institutions/domains, and online research.\n"
            "- 'doc_agent': Document and table parsing.\n"
            "- 'rag_agent': Internal knowledge base search.\n\n"
            "Delegation Principles:\n"
            "- Delegate the high-level GOAL/INTENT to the specialist agent. Specialist agents know their domain tools and will autonomously retrieve candidates, verify data, and execute operations.\n"
            "- For email organization, sorting, or moving: assign a single step to 'mail_agent' with the high-level goal. Do NOT emit micro-steps like 'create_label' or 'create_folder'.\n"
            "- If web verification or online research is needed, assign steps to 'web_agent'.\n"
            "- DAG Dependencies & Variable Passing:\n"
            "  * If a step depends on an earlier step, declare 'dependencies': [step_id, ...].\n"
            "  * Reference outputs from earlier steps using variable tokens in 'input_data':\n"
            "    - '$step_1.artifacts' or '$step_1.saved_paths' (for files downloaded/created in step 1)\n"
            "    - '$step_1.data' or '$step_1.<key>' (for extracted dictionary fields from step 1)\n\n"
            "Output strictly valid JSON in this exact structure:\n"
            "```json\n"
            "{\n"
            '  "goal": "<user request>",\n'
            '  "steps": [\n'
            "    {\n"
            '      "step_id": 1,\n'
            '      "assigned_agent": "<agent_name>",\n'
            '      "description": "<concise description of mission>",\n'
            '      "dependencies": [],\n'
            '      "input_data": {\n'
            '        "goal": "<high-level objective or specific intent>"\n'
            "      }\n"
            "    },\n"
            "    {\n"
            '      "step_id": 2,\n'
            '      "assigned_agent": "doc_agent",\n'
            '      "description": "<parse documents from step 1>",\n'
            '      "dependencies": [1],\n'
            '      "input_data": {\n'
            '        "instruction": "parse_document",\n'
            '        "file_path": "$step_1.artifacts"\n'
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )

        user_prompt = f"User Request:\n'{goal}'\n\nCognitive Context & Memory:\n{ctx_text}\n\nFormulate the execution plan now."

        tracer = get_tracer()
        if not self.client or not hasattr(self.client, "generate"):
            logger.warning("No active LLM client configured, building fallback plan.")
            plan = self._build_fallback_plan(goal, reason="No active LLM client configured")
            tracer.log_plan(
                "planner",
                goal,
                [{"step_id": s.step_id, "desc": s.description} for s in plan.steps],
                fallback=True,
                reason="No LLM client",
            )
            return plan

        try:
            response_text = self.client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.config.model_name,
                format="json",
            )
        except Exception as e:
            tracer.log_error("planner", e, context={"goal": goal})
            raise RuntimeError(f"Ollama plan generation failed: {e}") from e

        if response_text:
            try:
                plan = self._parse_plan_json(response_text, fallback_goal=goal)
                tracer.log_plan(
                    "planner",
                    goal,
                    [
                        {"step_id": s.step_id, "desc": s.description, "agent": s.assigned_agent}
                        for s in plan.steps
                    ],
                    fallback=False,
                )
                return plan
            except Exception as e:
                logger.warning(f"Falling back to heuristic plan after malformed planner JSON: {e}")
                plan = self._build_fallback_plan(goal, reason=str(e))
                tracer.log_plan(
                    "planner",
                    goal,
                    [
                        {"step_id": s.step_id, "desc": s.description, "agent": s.assigned_agent}
                        for s in plan.steps
                    ],
                    fallback=True,
                    reason=str(e),
                )
                return plan

        logger.warning("Ollama returned empty plan response, building fallback plan.")
        plan = self._build_fallback_plan(goal, reason="Empty LLM response")
        tracer.log_plan(
            "planner",
            goal,
            [{"step_id": s.step_id, "desc": s.description} for s in plan.steps],
            fallback=True,
            reason="Empty response",
        )
        return plan

    def generate_plan(self, goal: str, context: str | None = None) -> ExecutionPlan:
        """
        Public method to generate a structured execution plan. Delegates to _generate_plan.
        """
        return self._generate_plan(goal, context=context)

    def _parse_plan_json(self, response_text: str, fallback_goal: str) -> ExecutionPlan:
        """Parses LLM JSON response into an ExecutionPlan object with multi-format resilience."""
        data = extract_and_parse_json(response_text)
        if data is None:
            # Emergency regex step extraction if JSON structure was broken
            recovered_steps = []
            for m_str in re.finditer(
                r'\{[^{}]*?"assigned_agent"\s*:\s*"[^"]+"[^{}]*?\}', response_text
            ):
                from workos_engine.llm_client import repair_json_string

                s_dict = repair_json_string(m_str.group(0))
                if isinstance(s_dict, dict):
                    recovered_steps.append(s_dict)
            if recovered_steps:
                data = {"goal": fallback_goal, "steps": recovered_steps}

        if data is None:
            raise PlanParseError(f"planner response was not valid JSON: {response_text[:300]}")
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
            raise PlanParseError(
                f"planner response had unsupported JSON root type: {type(data).__name__}"
            )
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
            deps: list[int] = []
            if isinstance(s, str):
                desc = s
                step_id = idx
                desc_lower = desc.lower()
                if any(w in desc_lower for w in ["email", "emails", "inbox", "mailbox"]):
                    assigned_agent = "mail_agent"
                elif any(
                    w in desc_lower for w in ["web", "online", "search", "google", "internet"]
                ):
                    assigned_agent = "web_agent"
                elif any(w in desc_lower for w in ["doc", "parse", "pdf", "table"]):
                    assigned_agent = "doc_agent"
                else:
                    assigned_agent = "rag_agent"
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

                # Normalize agent name if needed
                desc_lower = desc.lower()
                if assigned_agent not in valid_agents:
                    if any(w in assigned_agent for w in ["mail", "email", "inbox"]):
                        assigned_agent = "mail_agent"
                    elif any(w in assigned_agent for w in ["doc", "parse", "pdf", "table"]):
                        assigned_agent = "doc_agent"
                    elif any(w in assigned_agent for w in ["rag", "knowledge", "index", "vector"]):
                        assigned_agent = "rag_agent"
                    elif any(
                        w in assigned_agent
                        for w in ["web", "online", "internet", "google", "browse"]
                    ):
                        assigned_agent = "web_agent"
                    elif any(w in desc_lower for w in ["email", "emails", "inbox", "mailbox"]):
                        assigned_agent = "mail_agent"
                    elif any(w in desc_lower for w in ["web", "online", "search online"]):
                        assigned_agent = "web_agent"
                    elif any(w in desc_lower for w in ["doc", "parse", "pdf", "table"]):
                        assigned_agent = "doc_agent"
                    else:
                        assigned_agent = "rag_agent"

                input_data = s.get("input_data") or s.get("params") or s.get("args") or {}
                if isinstance(input_data, str):
                    input_data = {"instruction": "execute", "query": input_data}
                elif not isinstance(input_data, dict):
                    input_data = {}

                if assigned_agent == "mail_agent" and any(
                    w in desc_lower for w in ["organize", "sort", "subfolder"]
                ):
                    if input_data.get("instruction") in (None, "", "execute"):
                        input_data["instruction"] = "organize_emails"

                raw_deps = s.get("dependencies") or s.get("depends_on") or []
                if isinstance(raw_deps, (int, str)):
                    try:
                        deps = [int(raw_deps)]
                    except ValueError:
                        deps = []
                elif isinstance(raw_deps, list):
                    deps = []
                    for d in raw_deps:
                        try:
                            deps.append(int(d))
                        except (ValueError, TypeError):
                            pass
                else:
                    deps = []
            else:
                continue

            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=desc,
                    assigned_agent=assigned_agent,
                    input_data=input_data,
                    dependencies=deps,
                    status="pending",
                )
            )

        if not steps:
            raise PlanParseError(
                f"Ollama returned plan JSON with no valid steps: {response_text[:300]}"
            )

        return ExecutionPlan(goal=goal, steps=steps)

    def _build_fallback_plan(self, goal: str, reason: str = "") -> ExecutionPlan:
        """Builds a conservative single-step plan when the planner model emits malformed JSON."""
        text = (goal or "").lower()
        if any(w in text for w in ["email", "emails", "inbox", "mailbox", "mail"]):
            assigned_agent = "mail_agent"
        elif any(
            w in text for w in ["web", "online", "internet", "google", "website", "url", "news"]
        ):
            assigned_agent = "web_agent"
        elif any(w in text for w in ["parse", "extract table", "pdf", "docx", "document file"]):
            assigned_agent = "doc_agent"
        else:
            assigned_agent = "rag_agent"

        input_data: dict[str, Any] = {
            "goal": goal,
            "query": goal,
            "planner_fallback": True,
        }
        if reason:
            input_data["planner_fallback_reason"] = reason[:500]

        return ExecutionPlan(
            goal=goal,
            steps=[
                PlanStep(
                    step_id=1,
                    description=f"Execute user request via {assigned_agent}",
                    assigned_agent=assigned_agent,
                    input_data=input_data,
                    status="pending",
                )
            ],
        )

    async def execute_plan(self, plan: ExecutionPlan, context: str | None = None) -> ExecutionPlan:
        """
        Executes each PlanStep in the ExecutionPlan in topological dependency order (DAG),
        resolving dynamic variables, tracking status transitions (pending -> in_progress -> completed / failed),
        and updating typed blackboard step outputs.
        """
        completed_steps: dict[int, PlanStep] = {}
        prev_step: PlanStep | None = None
        blackboard = BlackboardState(root_goal=plan.goal)

        # Build dependency graph
        graph: dict[int, set[int]] = {}
        step_map: dict[int, PlanStep] = {s.step_id: s for s in plan.steps}

        for step in plan.steps:
            deps = set(step.dependencies)
            # Detect implicit $step_X references in input_data
            for val in str(step.input_data).split():
                matches = re.findall(r"\$step_(\d+)", val)
                for m in matches:
                    deps.add(int(m))
            graph[step.step_id] = {d for d in deps if d in step_map and d != step.step_id}

        # Validate DAG and obtain topological execution order
        try:
            ts = TopologicalSorter(graph)
            ordered_step_ids = list(ts.static_order())
        except CycleError as e:
            logger.error(f"Cycle detected in execution plan dependencies: {e}")
            raise RuntimeError(f"Plan execution failed: cyclic dependency detected: {e}")

        for step_id in ordered_step_ids:
            step = step_map[step_id]

            # Check if any prerequisite failed
            failed_deps = [
                d for d in graph.get(step_id, set())
                if step_map[d].status != "completed"
            ]
            if failed_deps:
                step.status = "failed"
                step.result = ExecutionResult(
                    task_id=str(step.step_id),
                    agent_name=step.assigned_agent,
                    success=False,
                    error=f"Prerequisite step(s) {failed_deps} failed or were not completed.",
                )
                blackboard.add_receipt(
                    step.step_id,
                    step.assigned_agent,
                    f"Skipped due to failed dependencies: {failed_deps}",
                )
                prev_step = step
                continue

            step.status = "in_progress"
            blackboard.current_step_index = step.step_id

            # 1. Resolve variable references
            resolved_inputs = _resolve_variables_in_dict(
                step.input_data, completed_steps, prev_step, blackboard
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
                blackboard.add_receipt(
                    step.step_id,
                    step.assigned_agent,
                    f"Failed: unknown agent {step.assigned_agent}",
                )
                prev_step = step
                continue

            # 3. Extract instruction and context
            goal_arg = resolved_inputs.get("goal") or plan.goal
            instruction = resolved_inputs.get("instruction") or goal_arg or step.description
            context_args = {
                k: v for k, v in resolved_inputs.items() if k not in ("instruction", "goal")
            }
            context_args["goal"] = goal_arg
            context_args["step_description"] = step.description
            if context:
                context_args["cognitive_context"] = context

            # Inject sequential blackboard state and previous step receipts
            context_args["blackboard"] = blackboard
            context_args["previous_steps"] = list(blackboard.completed_steps)

            task = SubagentTask(
                task_id=f"step_{step.step_id}_{uuid.uuid4().hex[:6]}",
                agent_name=step.assigned_agent,
                instruction=instruction,
                context=context_args,
            )

            # 4. Dispatch to subagent
            step_t0 = datetime.now()
            try:
                if inspect.iscoroutinefunction(agent.execute):
                    res = await agent.execute(task)
                else:
                    res = agent.execute(task)
                    if inspect.iscoroutine(res):
                        res = await res
                step_dur_ms = int((datetime.now() - step_t0).total_seconds() * 1000)
                res.duration_ms = step_dur_ms
                step.duration_ms = step_dur_ms
                step.result = res
                step.status = "completed" if res.success else "failed"

                # Record receipt and typed data into blackboard
                summary = ""
                step_data = {}
                if isinstance(res.data, dict):
                    summary = (
                        res.data.get("receipt")
                        or res.data.get("message")
                        or f"{step.assigned_agent} completed {step.description}"
                    )
                    step_data = res.data
                else:
                    summary = f"{step.assigned_agent} completed {step.description} (status: {step.status})"
                    if res.data is not None:
                        step_data = {"data": res.data}

                if res.artifacts:
                    step_data["artifacts"] = res.artifacts
                    if "saved_path" not in step_data:
                        step_data["saved_path"] = res.artifacts[0]

                step.output_data = step_data
                blackboard.add_receipt(step.step_id, step.assigned_agent, summary, step_data)
                completed_steps[step.step_id] = step
            except Exception as e:
                logger.exception(f"Error executing step {step.step_id}: {e}")
                step.status = "failed"
                step.result = ExecutionResult(
                    task_id=task.task_id,
                    agent_name=step.assigned_agent,
                    success=False,
                    error=str(e),
                )
                blackboard.add_receipt(
                    step.step_id, step.assigned_agent, f"Failed with exception: {e}"
                )
            finally:
                prev_step = step

        return plan

    def synthesize_response(self, plan: ExecutionPlan, session_id: str | None = None) -> str:
        """
        Synthesizes a grounded final user-facing response from step execution results,
        taking recent conversational dialogue context into account.
        """
        step_summaries = []
        for s in plan.steps:
            status_str = (
                f"Step {s.step_id} ({s.assigned_agent}): {s.description} [{s.status.upper()}]"
            )
            if s.result:
                if s.result.success:
                    if isinstance(s.result.data, dict):
                        findings = s.result.data.get("findings") or s.result.data.get("summary")
                        steps_log = s.result.data.get("steps") or s.result.data.get("timeline")
                        if findings:
                            status_str += f"\nFindings: {findings}"
                        if steps_log:
                            status_str += "\nObserved Details:\n" + "\n".join(
                                f"  - {st}" for st in steps_log
                            )
                        if "organized_count" in s.result.data:
                            count = s.result.data.get("organized_count", 0)
                            folders = s.result.data.get("folders_created", [])
                            cat = s.result.data.get("category", "")
                            if count == 0:
                                status_str += f"\nOrganization Result: Zero matching emails found in inbox for '{cat}'. No emails were moved or folders created."
                            else:
                                status_str += (
                                    f"\nOrganization Result: Organized {count} emails into '{cat}'."
                                )
                                if folders:
                                    status_str += f"\nSubfolders Created/Used: {', '.join(folders)}"
                                actions = s.result.data.get("actions_taken", [])
                                if actions:
                                    status_str += "\nSample Organized Emails:\n" + "\n".join(
                                        f"  - [{a.get('label') or 'Email'}] {a.get('from') or 'Unknown Sender'}: {a.get('subject') or 'No Subject'} -> {a.get('destination')}"
                                        for a in actions[:15]
                                    )
                    else:
                        status_str += f"\nData: {str(s.result.data)[:1000]}"

                    if s.result.artifacts:
                        status_str += f"\nArtifacts: {s.result.artifacts}"
                else:
                    status_str += f"\nError: {s.result.error}"
            step_summaries.append(status_str)

        steps_text = "\n\n".join(step_summaries)

        if not self.client or not hasattr(self.client, "generate"):
            raise RuntimeError("No active LLM client configured for ExecutivePlanner synthesis.")

        session_context = ""
        if session_id:
            try:
                turns = self.memory.get_session_dialogue(session_id=session_id, limit=6)
                if turns:
                    dialogue_lines = [f"  [{t.role.upper()}]: {t.content}" for t in turns]
                    session_context = (
                        "Recent Conversation Context:\n" + "\n".join(dialogue_lines) + "\n\n"
                    )
            except Exception as e:
                logger.debug(f"Error getting session turns for synthesis: {e}")

        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            prompt = (
                f"You are the WorkOS Executive AI Operating System.\n"
                f"Current System Date: {today_str}\n\n"
                f"{session_context}"
                f"The user goal was: '{plan.goal}'\n\n"
                f"Step Execution History & Retrieved Findings:\n{steps_text}\n\n"
                f"Synthesize a clear, strictly grounded, professional executive brief based on the data above:\n"
                f"1. ZERO-HALLUCINATION: You MUST base 100% of your facts, tables, entity names, sender addresses, filenames, and numbers exclusively on the Step Execution History above.\n"
                f"2. If retrieved search results or email lists are empty (`[]`) or no matching records exist, you MUST explicitly state that no matching emails or records were found in the inbox. You are strictly FORBIDDEN from inventing fake filenames (e.g. email1.txt), placeholder accounts, or fake numbers.\n"
                f"3. When real emails are retrieved, cite real sender addresses, subject lines, dates, and whether they represent replies or outbound messages.\n"
                f"4. If multiple sources or webpages present differing perspectives, compare the arguments and summarize consensus.\n"
                f"5. Cite sources using [1], [2] referencing source URLs or documents.\n"
                f"6. Format mathematical expressions with LaTeX ($...$ or $$...$$) and code with markdown code fences.\n"
                f"7. COMPLETE BRIEF: Write a complete, thorough briefing with full explanatory sentences. Explain what actions were performed, what findings were discovered, and what results were achieved. Never stop after a title, header, or '---' divider line.\n"
                f"8. HUMAN-IN-THE-LOOP CLARIFICATIONS: If any decisions, status determinations, or critical details are missing, ambiguous, or undetermined from emails and documents (i.e. 'unknown' status), clearly list them under a dedicated '### Pending User Clarification / Unknown Decisions' section with concise questions for the user."
            )
            response_text = self.client.generate(
                prompt=prompt,
                model=self.config.model_name,
                temperature=getattr(self.config, "default_temperature", 0.6),
            )
            if response_text:
                summary = response_text.strip()
                # Clean up any trailing divider lines
                if summary.endswith("---"):
                    summary = summary[:-3].strip()
                # If the model still generated only a header, append the grounded findings
                if len(summary.splitlines()) <= 4 and steps_text:
                    summary += "\n\n**Executive Findings & Status:**\n" + "\n".join(
                        f"- {line}" for line in steps_text.splitlines() if line.strip()
                    )
                plan.final_output = summary
                return summary
            logger.warning(
                "Ollama returned empty synthesis response, generating grounded fallback."
            )
            fallback = f"### Executive Operations Brief\n\n**Goal:** {plan.goal}\n\n"
            if steps_text:
                fallback += f"**Step Execution History & Retrieved Findings:**\n{steps_text}\n"
            plan.final_output = fallback
            return fallback
        except Exception as e:
            logger.warning(
                f"Ollama synthesis generation failed ({e}), using grounded execution fallback."
            )
            fallback = f"### Executive Operations Brief\n\n**Goal:** {plan.goal}\n\n"
            if steps_text:
                fallback += f"**Step Execution History & Retrieved Findings:**\n{steps_text}\n"
            plan.final_output = fallback
            return fallback

    # Alias for compatibility with internal callers
    synthesize_output = synthesize_response

    async def reflect_async(
        self, goal: str, plan: ExecutionPlan, session_id: str | None = None
    ) -> None:
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

            # 2. Invoke reflector logic with conversation events
            events: list[Any] = [
                {"type": "user_goal", "content": goal, "role": "user"},
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
                {"type": "final_output", "content": plan.final_output or "", "role": "assistant"},
            ]

            # If session dialogue exists, prepend recent dialogue turns for full conversation awareness
            if session_id:
                try:
                    turns = self.memory.get_session_dialogue(session_id=session_id, limit=6)
                    dialogue_events = [{"role": t.role, "content": t.content} for t in turns]
                    events = dialogue_events + events
                except Exception:
                    pass

            self.memory.reflect_and_update(
                conversation_events=events,
                model_client=self.client,
            )
        except Exception as e:
            logger.warning(f"Async memory reflection error: {e}")

    # Alias for compatibility with internal callers
    async_reflect = reflect_async

    async def run_goal(
        self,
        goal: str,
        user_id: str = "default_user",
        session_id: str = "default_session",
        record_user_turn: bool = True,
    ) -> ExecutionPlan:
        """
        Receives user goal, maintains session dialogue, injects cognitive memory context,
        dynamically constructs structured execution plans, dispatches steps to specialist agents,
        synthesizes final outputs, and triggers background memory reflection.
        """
        # 0. Record user turn in session drawer
        if record_user_turn:
            self.memory.add_turn(session_id=session_id, role="user", content=goal)

        # 1. Inject cognitive memory context with session dialogue
        context = self.build_planning_context(goal, session_id=session_id)

        # 2. Generate structured execution plan
        plan = self.generate_plan(goal, context=context)

        # 3. Execute plan steps across specialist agents
        executed_plan = await self.execute_plan(plan, context=context)

        # 4. Synthesize final grounded response
        output = self.synthesize_response(executed_plan, session_id=session_id)

        # 5. Record assistant response in session drawer
        self.memory.add_turn(session_id=session_id, role="assistant", content=output)

        # 6. Trigger async reflection
        await self.reflect_async(goal, executed_plan, session_id=session_id)

        return executed_plan

    # Alias for compatibility with internal callers
    plan_and_execute = run_goal

    def route_intent(self, user_input: str, session_id: str | None = None) -> str:
        """
        Intelligently routes incoming user interaction:
        - 'CONVERSATIONAL': Inquiries about user profile, memory beliefs, past conversations,
          general assistant questions, advice, drafting, or greetings.
        - 'AGENT_EXECUTION': Operational workflows requiring multi-agent delegation (emails, documents, web, RAG).
        """
        text = (user_input or "").strip().lower()
        if not text:
            return "CONVERSATIONAL"
        if len(text.split()) <= 2 and not re.search(
            r"\b(email|emails|inbox|mail|web|search|find|parse|pdf|docx|document|rag|vault|send|move|organize|sort)\b",
            text,
        ):
            return "CONVERSATIONAL"

        # Explicit action keywords indicating tool execution
        action_patterns = [
            r"\b(organize|sort|label|move|download\s+attachment)\b.*(email|inbox|mail)",
            r"\b(send|draft|forward)\b.*(email|mail|message)\s+to\b",
            r"\b(parse|convert|extract\s+table)\b.*(pdf|doc|document|file)",
            r"\b(search|find|query)\b.*(web|internet|online|searxng|google)",
            r"\b(query|search|ask)\b.*(knowledge\s+base|rag|documents|index)",
            r"^(search|find|look\s+up)\s+",
        ]
        for pat in action_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return "AGENT_EXECUTION"

        conversational_patterns = [
            r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)\b",
            r"^(who\s+are\s+you|what\s+can\s+you\s+do|tell\s+me\s+about\s+yourself)\b",
            r"\b(where\s+(am\s+i|was\s+i)\s+(planning|applying|going))\b",
            r"\b(what\s+(are\s+my|is\s+my)\s+(preferences?|beliefs?|targets?|profile))\b",
            r"\b(what\s+do\s+you\s+know\s+about\s+me|remind\s+me)\b",
            r"\b(what\s+did\s+we\s+(talk|discuss)\s+about)\b",
            r"\b(help\s+me\s+(think|brainstorm|draft|write))\b",
            r"^(thanks?|thank\s+you|awesome|great|cool)\b",
        ]
        for pat in conversational_patterns:
            if re.search(pat, text, re.IGNORECASE):
                return "CONVERSATIONAL"

        # Short direct questions about self/knowledge
        words = text.split()
        if len(words) <= 10 and any(
            w in words for w in ["i", "my", "me", "you", "who", "where", "what", "how", "am", "was"]
        ):
            return "CONVERSATIONAL"

        return "AGENT_EXECUTION"

    async def chat_turn(
        self,
        message: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
    ) -> dict[str, Any]:
        """
        Interactive conversational assistant turn:
        Preserves dialogue history in MemPalace drawer, grounds answers in Hindsight cognitive memory,
        and dynamically dispatches specialist agent DAGs when tools are required.
        """
        tracer = get_tracer()
        try:
            # 0. Record user turn in session drawer
            self.memory.add_turn(session_id=session_id, role="user", content=message)

            intent = self.route_intent(message, session_id=session_id)
            tracer.record(
                event_type="CHAT_INTENT",
                component="planner",
                message=f"User prompt routed to {intent}",
                session_id=session_id,
                payload={"message": message, "intent": intent},
            )

            if intent == "AGENT_EXECUTION":
                t_start = datetime.now()
                plan = await self.run_goal(
                    message,
                    user_id=user_id,
                    session_id=session_id,
                    record_user_turn=False,
                )
                total_duration_ms = int((datetime.now() - t_start).total_seconds() * 1000)

                # Aggregate citations produced across RAG, Web research, and synthesis
                citations: list[dict[str, Any]] = []
                for step in plan.steps:
                    if step.result and isinstance(step.result.data, dict):
                        if "citations" in step.result.data and isinstance(step.result.data["citations"], list):
                            for cit in step.result.data["citations"]:
                                if isinstance(cit, dict) and not any(
                                    c.get("citation_id") == cit.get("citation_id") for c in citations
                                ):
                                    citations.append(cit)
                    elif step.result and isinstance(step.result.data, list):
                        for idx, r in enumerate(step.result.data, 1):
                            if isinstance(r, dict) and (r.get("url") or r.get("title")):
                                cit_id = f"[{idx}]"
                                if not any(c.get("citation_id") == cit_id for c in citations):
                                    citations.append(
                                        {
                                            "citation_id": cit_id,
                                            "title": r.get("title", f"Source {idx}"),
                                            "filename": r.get("title", f"Source {idx}"),
                                            "url": r.get("url", ""),
                                            "snippet": r.get("snippet", ""),
                                            "score": r.get("score", 1.0),
                                        }
                                    )
                    if step.result and isinstance(step.result.artifacts, list):
                        for art in step.result.artifacts:
                            if isinstance(art, dict) and "citation_id" in art:
                                if not any(c.get("citation_id") == art.get("citation_id") for c in citations):
                                    citations.append(art)

                # Also parse citations from final output if present
                if plan.final_output:
                    for m in re.finditer(r"\[(\d+)\]\s*([^\n\r]+)", plan.final_output):
                        cit_id = f"[{m.group(1)}]"
                        if not any(c.get("citation_id") == cit_id for c in citations):
                            line_content = m.group(2).strip()
                            url_match = re.search(r"https?://[^\s\)]+", line_content)
                            url = url_match.group(0) if url_match else ""
                            title = line_content.replace(url, "").strip(" -–—:") if url else line_content
                            citations.append(
                                {
                                    "citation_id": cit_id,
                                    "title": title or line_content,
                                    "filename": title or line_content,
                                    "url": url,
                                    "snippet": line_content,
                                    "score": 1.0,
                                }
                            )

                return {
                    "type": "plan_execution",
                    "message": plan.final_output or "Executed goal across specialist agents.",
                    "duration_ms": total_duration_ms,
                    "citations": citations,
                    "plan": {
                        "goal": plan.goal,
                        "final_output": plan.final_output,
                        "status": "completed"
                        if all(s.status == "completed" for s in plan.steps)
                        else "completed_with_issues",
                        "steps": [
                            {
                                "step_id": s.step_id,
                                "description": s.description,
                                "assigned_agent": s.assigned_agent,
                                "status": s.status,
                                "duration_ms": getattr(s, "duration_ms", None)
                                or (s.result.duration_ms if s.result else None),
                                "result": {
                                    "success": s.result.success if s.result else False,
                                    "data": s.result.data if s.result else None,
                                    "error": s.result.error if s.result else None,
                                    "artifacts": s.result.artifacts if s.result else [],
                                    "duration_ms": s.result.duration_ms if s.result else None,
                                }
                                if s.result
                                else None,
                            }
                            for s in plan.steps
                        ],
                    },
                    "session_id": session_id,
                }

            # Conversational / Assistant branch with deep cognitive grounding
            t_conv_start = datetime.now()
            cognitive_context = self.build_planning_context(message, session_id=session_id)
            today_str = datetime.now().strftime("%Y-%m-%d")

            system_prompt = (
                "You are Argus OS, an autonomous executive AI operating system and intelligent assistant.\n"
                f"Current System Date: {today_str}\n\n"
                "Persona & Behavioral Rules:\n"
                "- Articulate, executive, highly capable, and attentive.\n"
                "- Ground all your answers strictly in the user's stored beliefs, preferences, and recalled knowledge provided in the context.\n"
                "- If the user asks about their preferences, background, past decisions, or stored knowledge, recall and answer accurately from memory.\n"
                "- If the user asks for multi-domain actions (searching email, parsing documents, scraping web), summarize and offer to run the workflow.\n"
                "- Keep conversational responses clean, readable, and well-structured with clear markdown formatting."
            )
            user_prompt = (
                f"User Message:\n'{message}'\n\n"
                f"Cognitive Context & Memory:\n{cognitive_context}\n\n"
                "Respond directly as the user's executive assistant:"
            )

            try:
                response = self.client.generate(
                    prompt=user_prompt,
                    system=system_prompt,
                    model=self.config.model_name,
                    temperature=getattr(self.config, "default_temperature", 0.6),
                )
                response = response.strip()
            except Exception as e:
                logger.warning(f"Conversational generation error: {e}")
                response = f"I'm here as your Argus OS executive assistant. (System note: {e})"

            conv_duration_ms = int((datetime.now() - t_conv_start).total_seconds() * 1000)

            # Record assistant response in session drawer
            self.memory.add_turn(session_id=session_id, role="assistant", content=response)

            # Trigger reflection
            try:
                events = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ]
                self.memory.reflect_and_update(events, model_client=self.client)
            except Exception as e:
                logger.debug(f"Chat reflection error: {e}")

            return {
                "type": "conversation",
                "message": response,
                "duration_ms": conv_duration_ms,
                "session_id": session_id,
            }
        except Exception as e:
            logger.exception(f"Unhandled error in chat_turn: {e}")
            tracer.log_error(
                "planner.chat_turn", e, context={"message": message, "session_id": session_id}
            )
            err_msg = f"I encountered an error executing this workflow: {e}. Event has been recorded in the Debug Inspector."
            try:
                self.memory.add_turn(session_id=session_id, role="assistant", content=err_msg)
            except Exception:
                pass
            return {
                "type": "error",
                "message": err_msg,
                "session_id": session_id,
                "error": str(e),
            }
