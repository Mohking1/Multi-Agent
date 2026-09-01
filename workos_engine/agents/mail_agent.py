"""Mail specialist subagent enforcing autonomy policies and dispatching email tools."""

import os
from typing import Any

from config import WorkOSConfig
from workos_engine.agents.base import BaseSubagent
from workos_engine.tools.mail_tools import MailToolKit
from workos_engine.types import AutonomyLevel, ExecutionResult, SubagentTask


class MailAgent(BaseSubagent):
    """Specialist subagent for searching, reading, drafting, sending, and organizing emails."""

    name: str = "mail_agent"
    description: str = (
        "Specialist subagent for email management via IMAP and SMTP, "
        "including policy-governed sending, draft staging, and inbox organization."
    )

    def __init__(
        self,
        config: WorkOSConfig | None = None,
        toolkit: MailToolKit | None = None,
        model_client: Any | None = None,
    ):
        super().__init__(config=config, model_client=model_client)
        self.toolkit = toolkit or MailToolKit(config=self.config)

    def execute_send(
        self,
        to_email: str | list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        html: bool = False,
        attachments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sends an email or stages it as a draft depending on autonomy policy and recipient whitelist."""

        recipients = [to_email] if isinstance(to_email, str) else list(to_email)
        trusted = [t.lower().strip() for t in self.config.trusted_recipients]

        all_trusted = len(recipients) > 0 and all(r.lower().strip() in trusted for r in recipients)

        # In SUPERVISED mode, un-whitelisted recipients must be staged as draft
        if self.config.autonomy_level == AutonomyLevel.SUPERVISED and not all_trusted:
            draft_res = self.toolkit.create_draft(
                to_email=to_email,
                subject=subject,
                body=body,
                cc=cc,
                bcc=bcc,
            )
            return {
                "status": "User confirmation required. Staged as draft.",
                "staged_as_draft": True,
                "requires_confirmation": True,
                "recipient": to_email,
                "subject": subject,
                "draft": draft_res,
            }

        # In FULL autonomy mode or when all recipients are whitelisted
        send_res = self.toolkit.send_smtp_email(
            to_email=to_email,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            html=html,
            attachments=attachments,
        )
        send_res["staged_as_draft"] = False
        send_res["requires_confirmation"] = False
        return send_res

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Dispatches an individual tool call for the MailAgent."""
        t = tool_name.lower().strip()
        if t in ("search_emails", "search"):
            return self.toolkit.search_emails(**args)
        elif t in ("fetch_email", "fetch", "read"):
            return self.toolkit.fetch_email(**args)
        elif t in ("download_attachment", "download"):
            path = self.toolkit.download_attachment(**args)
            return {"saved_path": path}
        elif t in ("create_draft", "draft"):
            return self.toolkit.create_draft(**args)
        elif t in ("send_email", "send", "execute_send"):
            return self.execute_send(**args)
        elif t in ("move_email", "move"):
            return self.toolkit.move_email(**args)
        elif t in ("mark_email", "mark"):
            return self.toolkit.mark_email(**args)
        elif t in ("organize_emails", "organize", "organize_folder"):
            return self.toolkit.organize_emails(**args)
        else:
            raise ValueError(f"Unknown mail tool: {tool_name}")

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a delegated email task instruction or runs autonomous micro-ReAct loop."""
        instruction = (task.instruction or "").lower().strip()
        ctx = dict(task.context or {})

        # Direct mapped tool calls
        mapped_tools = {
            "search_emails": "search_emails",
            "search": "search_emails",
            "fetch_email": "fetch_email",
            "fetch": "fetch_email",
            "read": "fetch_email",
            "download_attachment": "download_attachment",
            "download": "download_attachment",
            "create_draft": "create_draft",
            "draft": "create_draft",
            "send_email": "send_email",
            "send": "send_email",
            "execute_send": "send_email",
            "move_email": "move_email",
            "move": "move_email",
            "mark_email": "mark_email",
            "mark": "mark_email",
            "organize_emails": "organize_emails",
            "organize": "organize_emails",
        }

        if instruction in mapped_tools:
            tool_name = mapped_tools[instruction]
            try:
                artifacts: list[str] = []
                data = self.execute_tool(tool_name, ctx)
                if (
                    isinstance(data, dict)
                    and "saved_path" in data
                    and os.path.exists(data["saved_path"])
                ):
                    artifacts.append(data["saved_path"])
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data=data,
                    artifacts=artifacts,
                )
            except Exception as e:
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=False,
                    error=str(e),
                )

        # Higher-level reasoning mission -> run micro-ReAct loop
        return super().execute(task)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns JSON schema definitions of all tools provided by the MailAgent."""
        return [
            {
                "name": "search_emails",
                "description": "Search emails in an IMAP folder using criteria such as sender, subject, date, seen status, and query text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sender": {
                            "type": "string",
                            "description": "Filter by sender email address.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Filter by keyword in email subject.",
                        },
                        "date_gte": {
                            "type": "string",
                            "description": "Filter emails on or after date (YYYY-MM-DD).",
                        },
                        "date_lt": {
                            "type": "string",
                            "description": "Filter emails before date (YYYY-MM-DD).",
                        },
                        "seen": {
                            "type": "boolean",
                            "description": "Filter read (true) or unread (false) emails.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "IMAP folder to search. Defaults to 'INBOX'.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of emails to return. Defaults to 20.",
                        },
                        "text": {
                            "type": "string",
                            "description": "Free text query across body and headers.",
                        },
                    },
                },
            },
            {
                "name": "fetch_email",
                "description": "Fetch full details, headers, body text, and attachment metadata of an email by UID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {
                            "type": "string",
                            "description": "Unique identifier of the email.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "IMAP folder name. Defaults to 'INBOX'.",
                        },
                        "mark_seen": {
                            "type": "boolean",
                            "description": "Whether to mark the email as read upon fetching. Defaults to false.",
                        },
                    },
                    "required": ["uid"],
                },
            },
            {
                "name": "download_attachment",
                "description": "Download a specific email attachment to a local directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {
                            "type": "string",
                            "description": "Unique identifier of the email containing the attachment.",
                        },
                        "attachment_filename": {
                            "type": "string",
                            "description": "Filename of the attachment to download.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "IMAP folder name. Defaults to 'INBOX'.",
                        },
                        "save_dir": {
                            "type": "string",
                            "description": "Local destination directory. Defaults to './downloads'.",
                        },
                    },
                    "required": ["uid", "attachment_filename"],
                },
            },
            {
                "name": "create_draft",
                "description": "Create and stage an email draft without sending.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_email": {
                            "type": "string",
                            "description": "Recipient email address.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Subject of the email.",
                        },
                        "body": {
                            "type": "string",
                            "description": "Body text of the email.",
                        },
                        "cc": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional CC recipients.",
                        },
                        "bcc": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional BCC recipients.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "Drafts folder name. Defaults to 'Drafts'.",
                        },
                    },
                    "required": ["to_email", "subject", "body"],
                },
            },
            {
                "name": "send_email",
                "description": "Send an email via SMTP. Autonomy policy gates may stage this as a draft if recipient is unverified.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_email": {
                            "type": "string",
                            "description": "Recipient email address.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Subject of the email.",
                        },
                        "body": {
                            "type": "string",
                            "description": "Body text or HTML of the email.",
                        },
                        "cc": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional CC recipients.",
                        },
                        "bcc": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional BCC recipients.",
                        },
                        "html": {
                            "type": "boolean",
                            "description": "Set to true if body content is HTML. Defaults to false.",
                        },
                    },
                    "required": ["to_email", "subject", "body"],
                },
            },
            {
                "name": "move_email",
                "description": "Move an email from one IMAP folder to another destination folder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {
                            "type": "string",
                            "description": "UID of the email to move.",
                        },
                        "destination_folder": {
                            "type": "string",
                            "description": "Destination IMAP folder (e.g. 'Archive', 'Invoices').",
                        },
                        "source_folder": {
                            "type": "string",
                            "description": "Source IMAP folder. Defaults to 'INBOX'.",
                        },
                    },
                    "required": ["uid", "destination_folder"],
                },
            },
            {
                "name": "mark_email",
                "description": "Update flags on an email (read/unread or starred/flagged).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {
                            "type": "string",
                            "description": "UID of the email to update.",
                        },
                        "seen": {
                            "type": "boolean",
                            "description": "Set read status (true for read, false for unread).",
                        },
                        "flagged": {
                            "type": "boolean",
                            "description": "Set flagged/starred status.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "IMAP folder name. Defaults to 'INBOX'.",
                        },
                    },
                    "required": ["uid"],
                },
            },
            {
                "name": "organize_emails",
                "description": "Automatically categorize, move, or flag emails matching specified rule criteria.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rules": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of rule dictionaries containing match conditions and actions.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "IMAP folder name to process. Defaults to 'INBOX'.",
                        },
                    },
                    "required": ["rules"],
                },
            },
        ]
