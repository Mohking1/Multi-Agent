"""IMAP, SMTP, and Email Organization ToolKit for WorkOS."""

import logging
import os
import smtplib
from datetime import date, datetime
from email.message import EmailMessage
from typing import Any

from imap_tools import AND, MailBox

from config import WorkOSConfig, get_config

logger = logging.getLogger(__name__)


class MailToolKit:
    """ToolKit providing IMAP searching/fetching/moving and SMTP sending/drafting operations."""

    def __init__(self, config: WorkOSConfig | None = None):
        self.config = config or get_config()

    def _get_mailbox(self, folder: str = "INBOX") -> MailBox:
        """Connects and logs into the IMAP mailbox on the given folder."""
        if not self.config.imap_host:
            raise ValueError("IMAP host is not configured.")
        mb = MailBox(self.config.imap_host, port=self.config.imap_port)
        return mb.login(self.config.imap_user, self.config.imap_password, initial_folder=folder)

    def search_emails(
        self,
        sender: str | None = None,
        subject: str | None = None,
        date_gte: Any | None = None,
        date_lt: Any | None = None,
        seen: bool | None = None,
        folder: str = "INBOX",
        limit: int = 25,
        text: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search emails in the specified IMAP folder matching filter criteria."""
        search_text = (text or query or "").strip()
        criteria: dict[str, Any] = {}
        if sender:
            criteria["from_"] = sender
        if subject:
            criteria["subject"] = subject
        if date_gte:
            criteria["date_gte"] = date_gte
        if date_lt:
            criteria["date_lt"] = date_lt
        if seen is not None:
            criteria["seen"] = seen
        if search_text:
            criteria["text"] = search_text

        results: list[dict[str, Any]] = []
        try:
            with self._get_mailbox(folder=folder) as mailbox:
                imap_query = AND(**criteria) if criteria else AND(all=True)
                messages = list(
                    mailbox.fetch(imap_query, limit=limit, reverse=True, mark_seen=False)
                )

                for msg in messages:
                    flags = list(getattr(msg, "flags", ()) or ())
                    is_seen = "\\Seen" in flags or "SEEN" in flags
                    attachments = getattr(msg, "attachments", []) or []
                    body_snippet = (
                        getattr(msg, "text", "") or getattr(msg, "html", "") or ""
                    ).strip()[:300]

                    msg_date = getattr(msg, "date", None)
                    date_str = (
                        msg_date.isoformat()
                        if isinstance(msg_date, (datetime, date))
                        else str(msg_date)
                        if msg_date
                        else None
                    )

                    results.append(
                        {
                            "uid": getattr(msg, "uid", ""),
                            "subject": getattr(msg, "subject", "") or "",
                            "from": getattr(msg, "from_", "") or "",
                            "to": list(getattr(msg, "to", ()) or ()),
                            "cc": list(getattr(msg, "cc", ()) or ()),
                            "bcc": list(getattr(msg, "bcc", ()) or ()),
                            "date": date_str,
                            "flags": flags,
                            "seen": is_seen,
                            "size": getattr(msg, "size", 0),
                            "has_attachments": len(attachments) > 0,
                            "snippet": body_snippet,
                        }
                    )
        except Exception as e:
            logger.warning(f"IMAP search error: {e}")

        return results

    def fetch_email(
        self, uid: str, folder: str = "INBOX", mark_seen: bool = False
    ) -> dict[str, Any]:
        """Fetch full email content, headers, and metadata by UID."""
        with self._get_mailbox(folder=folder) as mailbox:
            messages = list(mailbox.fetch(AND(uid=uid), mark_seen=mark_seen))
            if not messages:
                raise ValueError(f"Email with UID '{uid}' not found in folder '{folder}'.")
            msg = messages[0]

            attachments_info = []
            for att in getattr(msg, "attachments", []) or []:
                payload = getattr(att, "payload", b"")
                attachments_info.append(
                    {
                        "filename": getattr(att, "filename", "unnamed"),
                        "content_type": getattr(att, "content_type", "application/octet-stream"),
                        "size": getattr(att, "size", len(payload) if payload else 0),
                    }
                )

            msg_date = getattr(msg, "date", None)
            date_str = (
                msg_date.isoformat()
                if isinstance(msg_date, (datetime, date))
                else str(msg_date)
                if msg_date
                else None
            )

            return {
                "uid": getattr(msg, "uid", uid),
                "subject": getattr(msg, "subject", "") or "",
                "from": getattr(msg, "from_", "") or "",
                "to": list(getattr(msg, "to", ()) or ()),
                "cc": list(getattr(msg, "cc", ()) or ()),
                "bcc": list(getattr(msg, "bcc", ()) or ()),
                "date": date_str,
                "text": getattr(msg, "text", "") or "",
                "html": getattr(msg, "html", "") or "",
                "flags": list(getattr(msg, "flags", ()) or ()),
                "attachments": attachments_info,
            }

    def download_attachment(
        self,
        uid: str,
        attachment_filename: str,
        folder: str = "INBOX",
        save_dir: str = "./downloads",
    ) -> str:
        """Download an attachment from an email by UID to a local directory."""
        os.makedirs(save_dir, exist_ok=True)
        with self._get_mailbox(folder=folder) as mailbox:
            messages = list(mailbox.fetch(AND(uid=uid), mark_seen=False))
            if not messages:
                raise ValueError(f"Email with UID '{uid}' not found in folder '{folder}'.")
            msg = messages[0]

            target_att = None
            for att in getattr(msg, "attachments", []) or []:
                if getattr(att, "filename", "").lower() == attachment_filename.lower():
                    target_att = att
                    break

            if target_att is None:
                raise FileNotFoundError(
                    f"Attachment '{attachment_filename}' not found in email UID '{uid}'."
                )

            actual_filename = getattr(target_att, "filename", attachment_filename)
            saved_path = os.path.join(save_dir, actual_filename)
            with open(saved_path, "wb") as f:
                f.write(getattr(target_att, "payload", b""))

            return saved_path

    def create_draft(
        self,
        to_email: str | list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        folder: str = "Drafts",
    ) -> dict[str, Any]:
        """Create and stage an email draft, optionally syncing to IMAP Drafts folder."""
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.config.smtp_user or self.config.imap_user or "workos@local"
        if isinstance(to_email, list):
            msg["To"] = ", ".join(to_email)
        else:
            msg["To"] = to_email

        if cc:
            msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
        if bcc:
            msg["Bcc"] = ", ".join(bcc) if isinstance(bcc, list) else bcc
        msg.set_content(body)

        try:
            with self._get_mailbox(folder=folder) as mailbox:
                mailbox.append(msg.as_bytes(), folder=folder)
        except Exception:
            # Non-blocking if IMAP server is not reachable or drafts folder doesn't exist
            pass

        return {
            "status": "draft_created",
            "to": to_email,
            "subject": subject,
            "body": body,
            "cc": cc,
            "bcc": bcc,
            "staged_as_draft": True,
        }

    def send_smtp_email(
        self,
        to_email: str | list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        html: bool = False,
        attachments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send an email immediately via SMTP."""
        msg = EmailMessage()
        msg["Subject"] = subject
        from_addr = self.config.smtp_user or self.config.imap_user or "workos@local"
        msg["From"] = from_addr

        if isinstance(to_email, list):
            msg["To"] = ", ".join(to_email)
        else:
            msg["To"] = to_email

        if cc:
            msg["Cc"] = ", ".join(cc) if isinstance(cc, list) else cc
        if bcc:
            msg["Bcc"] = ", ".join(bcc) if isinstance(bcc, list) else bcc

        if html:
            msg.add_alternative(body, subtype="html")
        else:
            msg.set_content(body)

        if attachments:
            for att_path in attachments:
                if os.path.exists(att_path):
                    with open(att_path, "rb") as f:
                        file_data = f.read()
                        filename = os.path.basename(att_path)
                    msg.add_attachment(
                        file_data,
                        maintype="application",
                        subtype="octet-stream",
                        filename=filename,
                    )

        if not self.config.smtp_host:
            raise ValueError("SMTP host is not configured.")

        if self.config.smtp_port == 465:
            server = smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port)
        else:
            server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)

        with server:
            if self.config.smtp_port != 465:
                server.starttls()
            if self.config.smtp_user and self.config.smtp_password:
                server.login(self.config.smtp_user, self.config.smtp_password)
            server.send_message(msg)

        return {
            "status": "sent",
            "to": to_email,
            "subject": subject,
            "message_id": msg.get("Message-ID", ""),
        }

    def move_email(
        self, uid: str, destination_folder: str, source_folder: str = "INBOX"
    ) -> dict[str, Any]:
        """Move an email from one folder to another."""
        with self._get_mailbox(folder=source_folder) as mailbox:
            mailbox.move(uid, destination_folder)
        return {
            "status": "moved",
            "uid": uid,
            "from_folder": source_folder,
            "to_folder": destination_folder,
        }

    def mark_email(
        self,
        uid: str,
        seen: bool | None = None,
        flagged: bool | None = None,
        folder: str = "INBOX",
    ) -> dict[str, Any]:
        """Update flags on an email (read/seen status, starred/flagged)."""
        with self._get_mailbox(folder=folder) as mailbox:
            if seen is not None:
                mailbox.flag(uid, ["\\Seen"], seen)
            if flagged is not None:
                mailbox.flag(uid, ["\\Flagged"], flagged)
        return {
            "status": "marked",
            "uid": uid,
            "seen": seen,
            "flagged": flagged,
            "folder": folder,
        }

    def organize_emails(self, rules: list[dict[str, Any]], folder: str = "INBOX") -> dict[str, Any]:
        """Iterate over emails and apply matching categorization and movement rules."""
        processed = 0
        actions_taken = []
        with self._get_mailbox(folder=folder) as mailbox:
            messages = list(mailbox.fetch(AND(all=True), mark_seen=False))
            for msg in messages:
                processed += 1
                for rule in rules:
                    match_criteria = rule.get("match", {})
                    matched = True

                    if "sender" in match_criteria:
                        sender_val = match_criteria["sender"].lower()
                        msg_sender = (getattr(msg, "from_", "") or "").lower()
                        if sender_val not in msg_sender:
                            matched = False

                    if matched and "subject" in match_criteria:
                        subject_val = match_criteria["subject"].lower()
                        msg_subject = (getattr(msg, "subject", "") or "").lower()
                        if subject_val not in msg_subject:
                            matched = False

                    if matched and "seen" in match_criteria:
                        flags = list(getattr(msg, "flags", ()) or ())
                        is_seen = "\\Seen" in flags or "SEEN" in flags
                        if match_criteria["seen"] != is_seen:
                            matched = False

                    if matched:
                        action = rule.get("action")
                        uid = getattr(msg, "uid", "")
                        subject = getattr(msg, "subject", "")

                        if action == "move":
                            dest = rule.get("destination", "Archive")
                            mailbox.move(uid, dest)
                            actions_taken.append(
                                {
                                    "uid": uid,
                                    "action": "move",
                                    "destination": dest,
                                    "subject": subject,
                                }
                            )
                        elif action == "mark_seen":
                            mailbox.flag(uid, ["\\Seen"], True)
                            actions_taken.append(
                                {
                                    "uid": uid,
                                    "action": "mark_seen",
                                    "subject": subject,
                                }
                            )
                        elif action == "mark_unseen":
                            mailbox.flag(uid, ["\\Seen"], False)
                            actions_taken.append(
                                {
                                    "uid": uid,
                                    "action": "mark_unseen",
                                    "subject": subject,
                                }
                            )
                        elif action == "mark_flagged":
                            mailbox.flag(uid, ["\\Flagged"], True)
                            actions_taken.append(
                                {
                                    "uid": uid,
                                    "action": "mark_flagged",
                                    "subject": subject,
                                }
                            )
                        break  # Apply first matching rule per email

        return {
            "status": "organized",
            "processed_count": processed,
            "actions_taken": actions_taken,
        }
