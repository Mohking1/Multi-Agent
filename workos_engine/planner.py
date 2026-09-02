"""Executive Planner & Subagent Orchestrator for WorkOS."""

import inspect
import json
import logging
import re
import uuid
from datetime import datetime
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
    MemoryNetwork,
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
        Gathers active beliefs and relevant recalled domain facts/entities
        to build a compact, clean context prompt for planning.
        """
        parts = [f"Current System Date: {datetime.now().strftime('%Y-%m-%d')}"]

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
                    f"- [{b.wing}/{b.hall}] {b.key}: {b.content[:150]}" for b in active_beliefs
                ]
                parts.append("User Beliefs & System Preferences:\n" + "\n".join(belief_lines))
        except Exception as e:
            logger.debug(f"Error getting active beliefs: {e}")

        # 2. Relevant recalled domain facts and entities (strictly excluding raw noisy execution logs)
        try:
            recalled = self.memory.recall(query=goal, limit=5)
            clean_facts = []
            for m in recalled:
                if m.network in (
                    MemoryNetwork.FACTS,
                    MemoryNetwork.ENTITIES,
                    MemoryNetwork.BELIEFS,
                ):
                    clean_facts.append(f"- [{m.network.value}:{m.key}] {m.content[:200]}")
            if clean_facts:
                parts.append("Relevant Recalled Knowledge:\n" + "\n".join(clean_facts))
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
            "- 'web_agent': Live web search, news, portals, and online timelines.\n"
            "- 'doc_agent': Document and table parsing.\n"
            "- 'rag_agent': Internal knowledge base search.\n\n"
            "Agent Selection Rules:\n"
            "- If the user mentions 'emails', 'inbox', 'folder', 'organize emails', or 'sort emails': ALWAYS use 'mail_agent'. Do NOT use 'web_agent' unless the user explicitly requested online web searching.\n"
            "- Use 'web_agent' ONLY if the user explicitly asks to search online, web, internet, or check external deadlines.\n\n"
            "Output strictly valid JSON in this exact structure:\n"
            "{\n"
            '  "goal": "<user request>",\n'
            '  "steps": [\n'
            '    {"step_id": 1, "assigned_agent": "<agent_name>", "description": "<concise description of action>", "input_data": {"instruction": "<action>", "query": "<search keyword or parameters>"}}\n'
            "  ]\n"
            "}"
        )

        user_prompt = f"User Request:\n'{goal}'\n\nCognitive Context & Memory:\n{ctx_text}\n\nFormulate the execution plan now."

        if not self.client or not hasattr(self.client, "generate"):
            raise RuntimeError("No active LLM client configured for ExecutivePlanner.")

        try:
            response_text = self.client.generate(
                prompt=user_prompt,
                system=system_prompt,
                model=self.config.model_name,
                format="json",
                temperature=0.1,
            )
            if response_text:
                return self._parse_plan_json(response_text, fallback_goal=goal)
            raise RuntimeError("Ollama returned empty plan response.")
        except Exception as e:
            logger.error(f"Ollama plan generation failed: {e}")
            raise RuntimeError(f"Ollama plan generation failed: {e}") from e

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

                # Normalize agent name based on description and assigned name
                desc_lower = desc.lower()
                if any(
                    w in desc_lower
                    for w in [
                        "inbox",
                        "mailbox",
                        "folder",
                        "subfolder",
                        "organize email",
                        "move email",
                    ]
                ):
                    assigned_agent = "mail_agent"
                elif assigned_agent not in valid_agents:
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
                    else:
                        assigned_agent = (
                            "web_agent"
                            if ("web" in desc_lower or "online" in desc_lower)
                            else "mail_agent"
                        )

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
            raise RuntimeError(
                f"Ollama returned plan JSON with no valid steps: {response_text[:300]}"
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
            context_args.setdefault("goal", plan.goal)
            context_args.setdefault("step_description", step.description)

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
                    if isinstance(s.result.data, dict):
                        findings = s.result.data.get("findings") or ""
                        steps_log = s.result.data.get("steps") or []
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
                            status_str += (
                                f"\nOrganization Result: Organized {count} emails into '{cat}'."
                            )
                            if folders:
                                status_str += f"\nSubfolders Created/Used: {', '.join(folders)}"
                            actions = s.result.data.get("actions_taken", [])
                            if actions:
                                status_str += "\nSample Organized Emails:\n" + "\n".join(
                                    f"  - [{a.get('label') or 'Email'}] {a.get('from')}: {a.get('subject')} -> {a.get('destination')}"
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

        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            prompt = (
                f"You are the WorkOS Executive AI Operating System.\n"
                f"Current System Date: {today_str}\n"
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
            raise RuntimeError("Ollama returned empty synthesis response.")
        except Exception as e:
            logger.error(f"Ollama synthesis failed: {e}")
            raise RuntimeError(f"Ollama synthesis failed: {e}") from e

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
