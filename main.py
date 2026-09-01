"""WorkOS Interactive Console & Executive CLI Entrypoint."""
import argparse
import asyncio
import inspect
import os
import shlex
import sys
from typing import Any, Optional

from rich.box import ROUNDED, SIMPLE
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.status import Status
from rich.table import Table
from rich.text import Text

from config import WorkOSConfig, get_config
from workos_engine import __version__
from workos_engine.memory.networks import CognitiveMemoryEngine
from workos_engine.planner import ExecutivePlanner, _resolve_variables_in_dict
from workos_engine.types import (
    AutonomyLevel,
    ExecutionPlan,
    ExecutionResult,
    MemoryItem,
    MemoryNetwork,
    PlanStep,
    SubagentTask,
)


class WorkOSApp:
    """
    WorkOS Executive Interactive Console Application.
    Provides Rich interactive REPL, slash command routing, structured plan visualization,
    live step execution tracking, human-in-the-loop confirmation, and memory inspection.
    """

    def __init__(
        self,
        config: Optional[WorkOSConfig] = None,
        planner: Optional[ExecutivePlanner] = None,
        console: Optional[Console] = None,
    ):
        self.config = config or get_config()
        self.console = console or Console()
        self.planner = planner or ExecutivePlanner(config=self.config)
        self.memory: CognitiveMemoryEngine = self.planner.memory
        self.running: bool = True

    def display_banner(self) -> None:
        """Renders the executive WorkOS banner and active system status."""
        title = Text("WorkOS — Personal Executive AI Operating System", style="bold cyan")
        subtitle = Text(f"v{__version__} | Autonomous Multi-Agent Orchestration & Cognitive Memory", style="dim white")

        status_table = Table.grid(padding=(0, 2))
        status_table.add_column(style="bold yellow", justify="right")
        status_table.add_column(style="white")
        status_table.add_column(style="bold yellow", justify="right")
        status_table.add_column(style="white")

        autonomy_style = "bold green" if self.config.autonomy_level == AutonomyLevel.FULL else "bold yellow"
        status_table.add_row("Ollama URL:", f"{self.config.ollama_base_url}", "Autonomy Policy:", f"[{autonomy_style}]{self.config.autonomy_level.value}[/]")
        status_table.add_row("Model:", f"{self.config.model_name}", "Embeddings:", f"{self.config.embedding_model}")
        status_table.add_row("Memory DB:", f"{self.config.memory_db_path}", "Elasticsearch:", f"{self.config.elasticsearch_url}")
        status_table.add_row("Subagents:", "mail_agent, doc_agent, rag_agent", "Trusted Recip:", f"{len(self.config.trusted_recipients)} configured")

        banner_panel = Panel(
            status_table,
            title=title,
            subtitle=subtitle,
            box=ROUNDED,
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(banner_panel)
        self.console.print("[dim]Type [bold cyan]/help[/bold cyan] for slash commands or enter any natural language goal to execute.[/dim]\n")

    def display_help(self) -> None:
        """Renders comprehensive help table for slash commands and natural language goals."""
        table = Table(title="WorkOS Slash Commands & Usage", box=ROUNDED, header_style="bold cyan")
        table.add_column("Command", style="bold yellow", width=22)
        table.add_column("Arguments", style="green", width=28)
        table.add_column("Description", style="white")

        table.add_row("/help", "", "Show this commands reference table")
        table.add_row("/models", "", "List local Ollama models")
        table.add_row("/model", "<model_name>", "Switch active Ollama model")
        table.add_row("/policy", "[FULL | SUPERVISED]", "View or switch system autonomy policy")
        table.add_row("/plan", "<goal>", "Preview dynamic execution plan without executing")
        table.add_row("/memory", "[summary | loci]", "Display spatial Loci memory architecture summary")
        table.add_row("/memory search", "<query>", "Search 4-network cognitive memory with BM25 FTS5")
        table.add_row("/memory beliefs", "[wing]", "List active user beliefs & behavioral rules")
        table.add_row("/memory add", "<wing> <hall> <k> <val>", "Retain a verified fact directly into memory")
        table.add_row("/mail status", "", "Check IMAP/SMTP connectivity and configuration")
        table.add_row("/mail search", "<query>", "Search mailbox for matching emails")
        table.add_row("/mail draft", "<to> <subject> <body>", "Create email draft in mailbox")
        table.add_row("/mail send", "<to> <subject> <body>", "Send email (subject to autonomy policy)")
        table.add_row("/doc parse", "<file_path>", "Parse document using Docling parser")
        table.add_row("/doc tables", "<file_path>", "Extract structured tables from document")
        table.add_row("/rag search", "<query>", "Perform hybrid RRF search across Elasticsearch")
        table.add_row("/rag ingest", "<file_path>", "Ingest PDF document into hybrid RAG index")
        table.add_row("/rag ask", "<query>", "Query RAG knowledge base with citation grounding")
        table.add_row("/clear", "", "Clear terminal console screen")
        table.add_row("/exit", "", "Exit WorkOS console (/quit, /q)")

        self.console.print(table)
        self.console.print("\n[dim]Natural Language Input: Simply type any request (e.g. 'Search emails for invoice from Arvind, parse the PDF, and tell me the total amount').[/dim]\n")

    def render_plan(self, plan: ExecutionPlan) -> None:
        """Renders an ExecutionPlan object as a structured Rich Table."""
        table = Table(title=f"Execution Plan: {plan.goal}", box=ROUNDED, header_style="bold magenta", expand=True)
        table.add_column("Step", style="bold cyan", width=6, justify="center")
        table.add_column("Agent", style="bold yellow", width=14)
        table.add_column("Description", style="white", overflow="fold")
        table.add_column("Parameters / Inputs", style="dim green", overflow="fold")
        table.add_column("Status", style="bold", width=12, justify="center")

        status_colors = {
            "pending": "[dim]PENDING[/dim]",
            "in_progress": "[yellow]RUNNING[/yellow]",
            "completed": "[green]✓ DONE[/green]",
            "failed": "[red]✗ FAILED[/red]",
        }

        for step in plan.steps:
            status_text = status_colors.get(step.status, step.status)
            inputs_str = ", ".join(f"{k}={v}" for k, v in step.input_data.items()) if step.input_data else "-"
            table.add_row(
                str(step.step_id),
                step.assigned_agent,
                step.description,
                inputs_str,
                status_text,
            )

        self.console.print(table)

    async def _dispatch_agent(self, agent: Any, task: SubagentTask) -> ExecutionResult:
        """Dispatches task to subagent supporting both async and sync implementations."""
        if inspect.iscoroutinefunction(agent.execute):
            return await agent.execute(task)
        res = agent.execute(task)
        if inspect.iscoroutine(res):
            return await res
        return res

    async def execute_goal(self, goal: str, plan_only: bool = False) -> ExecutionPlan:
        """
        Generates and executes an ExecutionPlan for a natural language goal.
        Renders progress indicators, step results, and grounded final synthesis.
        """
        self.console.print(f"\n[bold cyan]✦ Goal:[/] [white]{goal}[/]")

        # 1. Plan generation
        with self.console.status("[bold cyan]Formulating executive plan...[/bold cyan]", spinner="dots"):
            context = self.planner.build_planning_context(goal)
            plan = self.planner.generate_plan(goal, context=context)

        self.render_plan(plan)

        if plan_only:
            self.console.print("[yellow]Plan-only mode: execution skipped.[/yellow]\n")
            return plan

        # 2. Execution phase with live status
        self.console.print("\n[bold cyan]✦ Executing Plan Steps:[/]")
        completed_steps: dict[int, PlanStep] = {}
        prev_step: Optional[PlanStep] = None

        for step in plan.steps:
            step.status = "in_progress"
            step_label = f"Step {step.step_id} [{step.assigned_agent}]: {step.description}"
            
            with self.console.status(f"[yellow]{step_label}...[/yellow]", spinner="dots"):
                step.input_data = _resolve_variables_in_dict(step.input_data, completed_steps, prev_step)
                
                agent = self.planner.agents.get(step.assigned_agent)
                if not agent:
                    step.status = "failed"
                    step.result = ExecutionResult(
                        task_id=str(step.step_id),
                        agent_name=step.assigned_agent,
                        success=False,
                        error=f"Unknown agent: {step.assigned_agent}",
                    )
                else:
                    instruction = step.input_data.get("instruction") or step.description
                    context_args = {k: v for k, v in step.input_data.items() if k != "instruction"}
                    task = SubagentTask(
                        task_id=f"step_{step.step_id}",
                        agent_name=step.assigned_agent,
                        instruction=instruction,
                        context=context_args,
                    )
                    try:
                        res = await self._dispatch_agent(agent, task)
                        step.result = res
                        step.status = "completed" if res.success else "failed"
                    except Exception as e:
                        step.status = "failed"
                        step.result = ExecutionResult(
                            task_id=task.task_id,
                            agent_name=step.assigned_agent,
                            success=False,
                            error=str(e),
                        )

            completed_steps[step.step_id] = step
            prev_step = step

            # Step outcome reporting
            if step.status == "completed":
                self.console.print(f"  [bold green]✓[/bold green] [white]Step {step.step_id}: {step.description}[/white]")
                if step.result and step.result.artifacts:
                    for art in step.result.artifacts:
                        self.console.print(f"    [dim cyan]Artifact saved:[/] [underline]{art}[/underline]")
            else:
                err = step.result.error if step.result else "Unknown error"
                self.console.print(f"  [bold red]✗[/bold red] [white]Step {step.step_id} Failed:[/] [red]{err}[/red]")

        # 3. Final synthesis
        with self.console.status("[bold cyan]Synthesizing final executive response...[/bold cyan]", spinner="dots"):
            final_summary = self.planner.synthesize_response(plan)

        # Render final output
        synthesis_panel = Panel(
            Markdown(final_summary) if "\n" in final_summary else Text(final_summary, style="white"),
            title="[bold green]WorkOS Executive Output[/bold green]",
            box=ROUNDED,
            border_style="green",
            padding=(1, 2),
        )
        self.console.print("\n", synthesis_panel, "\n")

        # 4. Async memory reflection
        try:
            await self.planner.reflect_async(goal, plan)
        except Exception as e:
            self.console.print(f"[dim red]Reflection warning: {e}[/dim red]")

        return plan

    async def handle_command(self, cmd_line: str) -> bool:
        """
        Routes user command line input.
        Returns False if application should terminate, True otherwise.
        """
        line = cmd_line.strip()
        if not line:
            return True

        # Slash commands
        if line.startswith("/"):
            try:
                tokens = shlex.split(line)
            except ValueError:
                tokens = line.split()

            if not tokens:
                return True

            cmd = tokens[0].lower()
            args = tokens[1:]

            if cmd in ("/exit", "/quit", "/q"):
                self.running = False
                self.console.print("[bold yellow]Exiting WorkOS. Goodbye![/bold yellow]")
                return False

            elif cmd == "/help":
                self.display_help()
                return True

            elif cmd == "/clear":
                self.console.clear()
                return True

            elif cmd in ("/model", "/models"):
                return await self._handle_model_cmd(args)

            elif cmd in ("/policy", "/autonomy"):
                return await self._handle_policy_cmd(args)

            elif cmd == "/plan":
                if not args:
                    self.console.print("[red]Usage: /plan <goal description>[/red]")
                    return True
                goal_str = line[len(cmd):].strip()
                await self.execute_goal(goal_str, plan_only=True)
                return True

            elif cmd == "/memory":
                return await self._handle_memory_cmd(args)

            elif cmd == "/mail":
                return await self._handle_mail_cmd(args)

            elif cmd == "/rag":
                return await self._handle_rag_cmd(args)

            elif cmd == "/doc":
                return await self._handle_doc_cmd(args)

            else:
                self.console.print(f"[red]Unknown slash command: {cmd}. Type /help for available commands.[/red]")
                return True

        # Natural language goal execution
        await self.execute_goal(line)
        return True

    async def _handle_model_cmd(self, args: list[str]) -> bool:
        """Handles /models and /model <name> commands for Ollama."""
        from workos_engine.llm_client import OllamaClient

        client = getattr(self.planner, "client", None)
        if not isinstance(client, OllamaClient):
            client = OllamaClient(
                base_url=self.config.ollama_base_url,
                default_model=self.config.model_name,
                embedding_model=self.config.embedding_model,
            )

        if not args:
            models = client.list_models()
            table = Table(title="Ollama Local Models", box=ROUNDED)
            table.add_column("Model Name", style="bold cyan")
            table.add_column("Status", style="yellow")

            if models:
                for m in models:
                    is_active = "[bold green]ACTIVE[/bold green]" if m == self.config.model_name else "available"
                    table.add_row(m, is_active)
            else:
                table.add_row("(No models returned from Ollama)", "[red]Check if Ollama is running[/red]")
            self.console.print(table)
            self.console.print(f"[dim]Current active model: [bold yellow]{self.config.model_name}[/bold yellow]. Use [cyan]/model <name>[/cyan] to switch.[/dim]")
            return True

        new_model = args[0]
        self.config.model_name = new_model
        self.planner.config.model_name = new_model
        if hasattr(self.planner, "client") and hasattr(self.planner.client, "default_model"):
            self.planner.client.default_model = new_model
        self.console.print(f"[green]✓ Active model switched to: [bold]{new_model}[/bold][/green]")
        return True

    async def _handle_policy_cmd(self, args: list[str]) -> bool:
        """Handles /policy [FULL | SUPERVISED] commands."""
        if not args:
            table = Table(title="Autonomy Policy Configuration", box=ROUNDED)
            table.add_column("Setting", style="bold yellow")
            table.add_column("Current Value", style="white")
            table.add_column("Description", style="dim")

            table.add_row(
                "Autonomy Level",
                f"[bold cyan]{self.config.autonomy_level.value}[/bold cyan]",
                "SUPERVISED requires approval for outbound actions; FULL executes autonomously.",
            )
            table.add_row(
                "Trusted Recipients",
                ", ".join(self.config.trusted_recipients) if self.config.trusted_recipients else "(none)",
                "Recipients allowed for automated email dispatch.",
            )
            self.console.print(table)
            return True

        level_arg = args[0].upper()
        if level_arg in ("FULL", "SUPERVISED"):
            new_level = AutonomyLevel(level_arg)
            self.config.autonomy_level = new_level
            self.planner.config.autonomy_level = new_level
            self.planner.mail_agent.config.autonomy_level = new_level
            self.console.print(f"[green]✓ Autonomy level updated to: [bold]{new_level.value}[/bold][/green]")
        else:
            self.console.print("[red]Invalid autonomy level. Choose either 'FULL' or 'SUPERVISED'.[/red]")
        return True

    async def _handle_memory_cmd(self, args: list[str]) -> bool:
        """Handles /memory subcommands."""
        if not args or args[0] in ("summary", "loci"):
            summary = self.memory.get_context_index_summary()
            panel = Panel(
                Text(summary or "No memories indexed yet.", style="cyan"),
                title="Spatial Loci Memory Structure",
                box=ROUNDED,
                border_style="magenta",
            )
            self.console.print(panel)
            return True

        subcmd = args[0].lower()

        if subcmd == "search":
            query = " ".join(args[1:]) if len(args) > 1 else ""
            if not query:
                self.console.print("[red]Usage: /memory search <query>[/red]")
                return True
            results = self.memory.recall(query=query, limit=10)
            if not results:
                self.console.print(f"[yellow]No memories found matching: '{query}'[/yellow]")
                return True

            table = Table(title=f"Memory Search Results: '{query}'", box=ROUNDED)
            table.add_column("Network", style="bold cyan", width=12)
            table.add_column("Locus (Wing/Hall)", style="yellow", width=22)
            table.add_column("Key", style="green", width=18)
            table.add_column("Content", style="white")

            for item in results:
                table.add_row(
                    item.network.value if hasattr(item.network, "value") else str(item.network),
                    f"{item.wing}/{item.hall}",
                    item.key,
                    item.content[:80] + ("..." if len(item.content) > 80 else ""),
                )
            self.console.print(table)
            return True

        elif subcmd == "beliefs":
            wing = args[1] if len(args) > 1 else None
            beliefs = self.memory.get_active_beliefs(wing=wing)
            if not beliefs:
                self.console.print("[yellow]No active beliefs recorded.[/yellow]")
                return True

            table = Table(title="Active Beliefs & Behavioral Rules", box=ROUNDED)
            table.add_column("Wing/Hall", style="yellow", width=22)
            table.add_column("Key", style="green", width=20)
            table.add_column("Content", style="white")

            for b in beliefs:
                table.add_row(f"{b.wing}/{b.hall}", b.key, b.content)
            self.console.print(table)
            return True

        elif subcmd == "add":
            if len(args) < 5:
                self.console.print("[red]Usage: /memory add <wing> <hall> <key> <content>[/red]")
                return True
            wing, hall, key = args[1], args[2], args[3]
            content = " ".join(args[4:])
            item_id = self.memory.retain_fact(wing=wing, hall=hall, key=key, content=content)
            self.console.print(f"[green]✓ Fact retained in memory (ID: {item_id}) under [{wing}/{hall}][/green]")
            return True

        else:
            self.console.print(f"[red]Unknown memory subcommand: {subcmd}. Use summary, search, beliefs, or add.[/red]")
            return True

    async def _handle_mail_cmd(self, args: list[str]) -> bool:
        """Handles /mail subcommands."""
        if not args or args[0] == "status":
            table = Table(title="Mail Subagent Status & Configuration", box=ROUNDED)
            table.add_column("Setting", style="bold yellow")
            table.add_column("Value", style="white")

            table.add_row("IMAP Host:", f"{self.config.imap_host or 'Not set'}:{self.config.imap_port}")
            table.add_row("IMAP User:", f"{self.config.imap_user or 'Not set'}")
            table.add_row("SMTP Host:", f"{self.config.smtp_host or 'Not set'}:{self.config.smtp_port}")
            table.add_row("SMTP User:", f"{self.config.smtp_user or 'Not set'}")
            table.add_row("Autonomy:", f"{self.config.autonomy_level.value}")
            self.console.print(table)
            return True

        subcmd = args[0].lower()
        if subcmd == "search":
            query = " ".join(args[1:]) if len(args) > 1 else ""
            task = SubagentTask("mail_cli", "mail_agent", "search_emails", {"text": query})
            res = await self._dispatch_agent(self.planner.mail_agent, task)
            if res.success:
                emails = res.data.get("emails", []) if isinstance(res.data, dict) else res.data
                self.console.print(f"[green]Found {len(emails)} matching emails.[/green]")
                for em in emails[:5]:
                    self.console.print(f"  • [bold]{em.get('subject', 'No Subject')}[/] from [cyan]{em.get('sender', '')}[/] ({em.get('date', '')})")
            else:
                self.console.print(f"[red]Mail search error: {res.error}[/red]")
            return True

        elif subcmd in ("draft", "send"):
            if len(args) < 4:
                self.console.print(f"[red]Usage: /mail {subcmd} <to> <subject> <body>[/red]")
                return True
            to_email, subject = args[1], args[2]
            body = " ".join(args[3:])
            instr = "send_email" if subcmd == "send" else "create_draft"
            task = SubagentTask("mail_cli", "mail_agent", instr, {"to_email": to_email, "subject": subject, "body": body})
            res = await self._dispatch_agent(self.planner.mail_agent, task)
            if res.success:
                self.console.print(f"[green]✓ Email {subcmd} action completed: {res.data}[/green]")
            else:
                self.console.print(f"[red]Mail action failed: {res.error}[/red]")
            return True

        else:
            self.console.print(f"[red]Unknown /mail subcommand: {subcmd}[/red]")
            return True

    async def _handle_rag_cmd(self, args: list[str]) -> bool:
        """Handles /rag subcommands."""
        if not args:
            self.console.print("[red]Usage: /rag [search <query> | ingest <file> | ask <query>][/red]")
            return True

        subcmd = args[0].lower()
        query_or_file = " ".join(args[1:]) if len(args) > 1 else ""

        if subcmd == "search":
            task = SubagentTask("rag_cli", "rag_agent", "rag_search", {"query": query_or_file})
            res = await self._dispatch_agent(self.planner.rag_agent, task)
            if res.success:
                hits = res.data if isinstance(res.data, list) else []
                self.console.print(f"[green]Found {len(hits)} relevant knowledge chunks:[/green]")
                for h in hits[:5]:
                    txt = h.get("text", "") if isinstance(h, dict) else str(h)
                    self.console.print(Panel(txt[:300], title=f"Score: {h.get('score', 'N/A') if isinstance(h, dict) else ''}", box=ROUNDED))
            else:
                self.console.print(f"[red]RAG search failed: {res.error}[/red]")
            return True

        elif subcmd == "ingest":
            task = SubagentTask("rag_cli", "rag_agent", "rag_ingest_pdf", {"file_path": query_or_file})
            res = await self._dispatch_agent(self.planner.rag_agent, task)
            if res.success:
                self.console.print(f"[green]✓ Document ingested into RAG: {res.data}[/green]")
            else:
                self.console.print(f"[red]RAG ingestion failed: {res.error}[/red]")
            return True

        elif subcmd == "ask":
            task = SubagentTask("rag_cli", "rag_agent", "rag_ask", {"query": query_or_file})
            res = await self._dispatch_agent(self.planner.rag_agent, task)
            if res.success:
                ans = res.data.get("answer", res.data) if isinstance(res.data, dict) else res.data
                self.console.print(Panel(str(ans), title="RAG Answer", box=ROUNDED, border_style="cyan"))
            else:
                self.console.print(f"[red]RAG ask failed: {res.error}[/red]")
            return True

        else:
            self.console.print(f"[red]Unknown /rag subcommand: {subcmd}[/red]")
            return True

    async def _handle_doc_cmd(self, args: list[str]) -> bool:
        """Handles /doc subcommands."""
        if not args:
            self.console.print("[red]Usage: /doc [parse <file> | tables <file>][/red]")
            return True

        subcmd = args[0].lower()
        file_path = " ".join(args[1:]) if len(args) > 1 else ""

        if subcmd == "parse":
            task = SubagentTask("doc_cli", "doc_agent", "parse_document", {"file_path": file_path})
            res = await self._dispatch_agent(self.planner.doc_agent, task)
            if res.success:
                self.console.print(f"[green]✓ Document parsed successfully.[/green]")
                txt = res.data.get("markdown", str(res.data)) if isinstance(res.data, dict) else str(res.data)
                self.console.print(Panel(txt[:500] + ("..." if len(txt) > 500 else ""), title=f"Doc Content: {file_path}", box=ROUNDED))
            else:
                self.console.print(f"[red]Doc parsing failed: {res.error}[/red]")
            return True

        elif subcmd == "tables":
            task = SubagentTask("doc_cli", "doc_agent", "extract_tables", {"file_path": file_path})
            res = await self._dispatch_agent(self.planner.doc_agent, task)
            if res.success:
                tables = res.data.get("tables", []) if isinstance(res.data, dict) else res.data
                self.console.print(f"[green]Extracted {len(tables)} tables.[/green]")
            else:
                self.console.print(f"[red]Table extraction failed: {res.error}[/red]")
            return True

        else:
            self.console.print(f"[red]Unknown /doc subcommand: {subcmd}[/red]")
            return True

    async def run_interactive(self) -> None:
        """Starts the interactive CLI prompt loop."""
        self.display_banner()

        while self.running:
            try:
                user_input = Prompt.ask("[bold cyan]WorkOS[/bold cyan][dim]>[/dim]")
                if not user_input or not user_input.strip():
                    continue

                keep_running = await self.handle_command(user_input)
                if not keep_running:
                    break
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[bold yellow]Exiting WorkOS. Goodbye![/bold yellow]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Unexpected error:[/] {e}")


def create_app(
    config: Optional[WorkOSConfig] = None,
    planner: Optional[ExecutivePlanner] = None,
    console: Optional[Console] = None,
) -> WorkOSApp:
    """Factory function to instantiate WorkOSApp."""
    return WorkOSApp(config=config, planner=planner, console=console)


def parse_cli_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="WorkOS — Personal Executive AI Operating System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--goal", "-g",
        type=str,
        default=None,
        help="Execute a natural language goal directly in non-interactive mode.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Generate and display execution plan without executing steps.",
    )
    parser.add_argument(
        "--autonomy", "-a",
        type=str,
        choices=["FULL", "SUPERVISED"],
        default=None,
        help="Override system autonomy policy level.",
    )
    parser.add_argument(
        "--memory-db",
        type=str,
        default=None,
        help="Path to SQLite cognitive memory database.",
    )
    parser.add_argument(
        "--help-commands",
        action="store_true",
        help="Display slash commands reference table and exit.",
    )
    return parser.parse_args(args)


async def main() -> None:
    """CLI application entrypoint."""
    cli_args = parse_cli_args()
    config = get_config()

    if cli_args.autonomy:
        config.autonomy_level = AutonomyLevel(cli_args.autonomy)
    if cli_args.memory_db:
        config.memory_db_path = cli_args.memory_db

    app = create_app(config=config)

    if cli_args.help_commands:
        app.display_help()
        return

    if cli_args.goal:
        await app.execute_goal(cli_args.goal, plan_only=cli_args.plan_only)
    else:
        await app.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
