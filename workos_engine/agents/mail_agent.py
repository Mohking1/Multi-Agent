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
    ):
        super().__init__(config=config)
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

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a delegated email task instruction."""
        instruction = (task.instruction or "").lower().strip()
        ctx = dict(task.context or {})

        try:
            artifacts: list[str] = []
            if instruction in ("search_emails", "search"):
                data = self.toolkit.search_emails(**ctx)
            elif instruction in ("fetch_email", "fetch", "read"):
                data = self.toolkit.fetch_email(**ctx)
            elif instruction in ("download_attachment", "download"):
                file_path = self.toolkit.download_attachment(**ctx)
                data = {"saved_path": file_path}
                if os.path.exists(file_path):
                    artifacts.append(file_path)
            elif instruction in ("create_draft", "draft"):
                data = self.toolkit.create_draft(**ctx)
            elif instruction in ("send_email", "send", "execute_send"):
                data = self.execute_send(**ctx)
            elif instruction in ("move_email", "move"):
                data = self.toolkit.move_email(**ctx)
            elif instruction in ("mark_email", "mark"):
                data = self.toolkit.mark_email(**ctx)
            elif instruction in ("organize_emails", "organize", "organize_folder"):
                data = self.toolkit.organize_emails(**ctx)
            else:
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=False,
                    error=f"Unknown instruction: {task.instruction}",
                )

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
