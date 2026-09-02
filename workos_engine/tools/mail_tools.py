"""IMAP, SMTP, and Email Organization ToolKit for WorkOS."""

import logging
import os
import re
import smtplib
import unicodedata
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

    def _parse_date(self, d: Any) -> date | None:
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            try:
                return datetime.fromisoformat(d.replace("Z", "+00:00")).date()
            except Exception:
                try:
                    return datetime.strptime(d[:10], "%Y-%m-%d").date()
                except Exception:
                    pass
        return None

    def _extract_domain_label(self, sender: str) -> str | None:
        """Extract a clean, human-readable organization or domain label from an email address generically."""
        if not sender or "@" not in sender:
            return None
        domain_part = sender.split("@")[-1].strip().rstrip(">").lower()
        parts = domain_part.split(".")
        if len(parts) >= 2:
            primary = parts[-2] if len(parts[-1]) <= 3 and len(parts) >= 2 else parts[0]
            clean_name = re.sub(r"[^a-zA-Z0-9_-]", "", primary).strip()
            if clean_name and len(clean_name) >= 2:
                return (
                    clean_name.upper()
                    if len(clean_name) <= 4
                    else clean_name.replace("-", " ").title()
                )
        return None

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
        def _clean_ascii(s: str) -> str:
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

        search_text = _clean_ascii((text or query or "").strip())
        criteria: dict[str, Any] = {}
        if sender:
            criteria["from_"] = _clean_ascii(sender)
        if subject:
            criteria["subject"] = _clean_ascii(subject)

        if date_gte:
            parsed_gte = self._parse_date(date_gte)
            if parsed_gte:
                criteria["date_gte"] = parsed_gte
        if date_lt:
            parsed_lt = self._parse_date(date_lt)
            if parsed_lt:
                criteria["date_lt"] = parsed_lt
        if seen is not None:
            criteria["seen"] = seen
        if search_text:
            criteria["text"] = search_text

        def _format_msg_dict(msg: Any) -> dict[str, Any]:
            flags = list(getattr(msg, "flags", ()) or ())
            is_seen = "\\Seen" in flags or "SEEN" in flags
            attachments = getattr(msg, "attachments", []) or []
            body_snippet = (getattr(msg, "text", "") or getattr(msg, "html", "") or "").strip()[
                :300
            ]
            msg_date = getattr(msg, "date", None)
            date_str = (
                msg_date.isoformat()
                if isinstance(msg_date, (datetime, date))
                else str(msg_date)
                if msg_date
                else None
            )
            return {
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

        results: list[dict[str, Any]] = []
        try:
            with self._get_mailbox(folder=folder) as mailbox:
                imap_query = AND(**criteria) if criteria else AND(all=True)
                messages = list(
                    mailbox.fetch(imap_query, limit=limit, reverse=True, mark_seen=False)
                )

                # If exact phrase match returned no results, search distinct keywords
                if not messages and search_text:
                    terms = [
                        w.strip() for w in re.split(r"[\s,]+", search_text) if len(w.strip()) >= 3
                    ]
                    if len(terms) > 1:
                        seen_uids = set()
                        for t in terms:
                            term_criteria = dict(criteria)
                            term_criteria["text"] = t
                            for msg in mailbox.fetch(
                                AND(**term_criteria), limit=limit, reverse=True, mark_seen=False
                            ):
                                uid = getattr(msg, "uid", "")
                                if uid not in seen_uids:
                                    seen_uids.add(uid)
                                    messages.append(msg)

                for msg in messages:
                    results.append(_format_msg_dict(msg))
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

    def list_folders(self) -> list[dict[str, Any]]:
        """List all available IMAP folders / labels and their hierarchy delimiter."""
        with self._get_mailbox() as mailbox:
            folder_list = []
            for f in mailbox.folder.list():
                folder_list.append(
                    {
                        "name": f.name,
                        "delimiter": getattr(f, "delim", "/") or "/",
                        "flags": list(getattr(f, "flags", ()) or ()),
                    }
                )
            return folder_list

    def create_folder(self, folder: str) -> dict[str, Any]:
        """
        Create a new folder or nested subfolder hierarchy (e.g. 'Archive/2026', 'Projects/Alpha').
        Sequentially ensures that each ancestor folder exists from root to leaf,
        supporting IMAP server delimiters (such as '/' in Gmail).
        """
        with self._get_mailbox() as mailbox:
            delim = "/"
            try:
                for f in mailbox.folder.list():
                    if getattr(f, "delim", None):
                        delim = f.delim
                        break
            except Exception:
                pass

            normalized_folder = folder.replace("\\", delim).replace("/", delim).strip(delim)
            parts = [p.strip() for p in normalized_folder.split(delim) if p.strip()]

            created_folders = []
            current_path = ""
            for part in parts:
                current_path = f"{current_path}{delim}{part}" if current_path else part
                try:
                    if not mailbox.folder.exists(current_path):
                        mailbox.folder.create(current_path)
                        created_folders.append(current_path)
                except Exception as e:
                    err_msg = str(e).lower()
                    if (
                        "exist" not in err_msg
                        and "duplicate" not in err_msg
                        and "already" not in err_msg
                    ):
                        raise

            return {
                "status": "created" if created_folders else "exists",
                "folder": current_path,
                "created_hierarchy": created_folders,
                "delimiter": delim,
            }

    def move_email(
        self, uid: str, destination_folder: str, source_folder: str = "INBOX"
    ) -> dict[str, Any]:
        """Move an email from one folder to another, ensuring the destination hierarchy exists."""
        self.create_folder(destination_folder)
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

    def organize_emails(
        self,
        rules: list[dict[str, Any]] | None = None,
        category: str | None = None,
        date_gte: Any = None,
        date_lt: Any = None,
        nested_subfolders: bool = True,
        domain_pattern: str | None = None,
        folder: str = "INBOX",
        limit: int = 500,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Organize emails in the mailbox.
        Supports:
        1. Category-based organization (e.g. category='Archive', nested_subfolders=True)
        2. Rule-based criteria (rules=[{"match": {...}, "action": "move", "destination": "..."}])
        """
        actions_taken = []
        processed = 0

        # Mode A: Category / Domain Organization
        if category or not rules:
            if not category and not domain_pattern and not kwargs.get("text") and not date_gte:
                return {
                    "status": "skipped",
                    "category": None,
                    "organized_count": 0,
                    "folders_created": [],
                    "message": "Organization skipped: No category, domain filter, or search criteria provided.",
                }
            root_folder = category or "Organized"
            self.create_folder(root_folder)

            criteria: dict[str, Any] = {}
            if date_gte:
                parsed_gte = self._parse_date(date_gte)
                if parsed_gte:
                    criteria["date_gte"] = parsed_gte
            if date_lt:
                parsed_lt = self._parse_date(date_lt)
                if parsed_lt:
                    criteria["date_lt"] = parsed_lt

            if domain_pattern:
                criteria["from_"] = domain_pattern

            text_filter = kwargs.get("text")
            if text_filter and isinstance(text_filter, str) and text_filter.strip():
                clean_q = (
                    unicodedata.normalize("NFKD", text_filter)
                    .encode("ascii", "ignore")
                    .decode("ascii")
                    .strip()
                )
                if clean_q:
                    criteria["text"] = clean_q

            folders_to_search = [folder]
            if folder == "INBOX":
                folders_to_search.append("[Gmail]/All Mail")

            folders_created = set()
            moved_uids = set()

            for search_folder in folders_to_search:
                try:
                    with self._get_mailbox(folder=search_folder) as mailbox:
                        delim = "/"
                        try:
                            for f in mailbox.folder.list():
                                if getattr(f, "delim", None):
                                    delim = f.delim
                                    break
                        except Exception:
                            pass

                        def _ensure_on_connection(mb: Any, path: str, d: str = "/") -> None:
                            norm = path.replace("\\", d).replace("/", d).strip(d)
                            parts = [p.strip() for p in norm.split(d) if p.strip()]
                            curr = ""
                            for p in parts:
                                curr = f"{curr}{d}{p}" if curr else p
                                try:
                                    if not mb.folder.exists(curr):
                                        mb.folder.create(curr)
                                except Exception as err:
                                    err_str = str(err).lower()
                                    if (
                                        "exist" not in err_str
                                        and "duplicate" not in err_str
                                        and "already" not in err_str
                                    ):
                                        logger.debug(f"Error creating '{curr}': {err}")

                        _ensure_on_connection(mailbox, root_folder, delim)

                        try:
                            messages = list(
                                mailbox.fetch(
                                    AND(**criteria) if criteria else AND(all=True),
                                    limit=limit,
                                    reverse=True,
                                    headers_only=True,
                                    mark_seen=False,
                                )
                            )
                        except Exception as e:
                            logger.warning(f"Error fetching emails from {search_folder}: {e}")
                            messages = []

                        for msg in messages:
                            processed += 1
                            uid = str(getattr(msg, "uid", ""))
                            if uid in moved_uids:
                                continue

                            sender = getattr(msg, "from_", "") or ""
                            subject = getattr(msg, "subject", "") or ""

                            if domain_pattern:
                                pat = domain_pattern.lower().strip().lstrip(".")
                                sender_domain = (
                                    sender.split("@")[-1].lower().rstrip(">")
                                    if "@" in sender
                                    else ""
                                )
                                if not (
                                    sender_domain.endswith(f".{pat}")
                                    or sender_domain == pat
                                    or f".{pat}." in sender_domain
                                ):
                                    continue

                            domain_label = self._extract_domain_label(sender)
                            if not domain_label:
                                continue

                            if domain_label.lower() in (
                                "gmail",
                                "yahoo",
                                "hotmail",
                                "outlook",
                                "icloud",
                            ):
                                continue

                            dest_folder = (
                                f"{root_folder}/{domain_label}"
                                if nested_subfolders
                                else root_folder
                            )
                            try:
                                _ensure_on_connection(mailbox, dest_folder, delim)
                                folders_created.add(dest_folder)
                            except Exception as e:
                                logger.debug(f"Ensuring folder '{dest_folder}': {e}")

                            try:
                                mailbox.move(uid, dest_folder)
                                moved_uids.add(uid)
                                if nested_subfolders and root_folder != dest_folder:
                                    try:
                                        mailbox.copy(uid, root_folder)
                                    except Exception:
                                        pass

                                actions_taken.append(
                                    {
                                        "uid": uid,
                                        "action": "move",
                                        "label": domain_label,
                                        "destination": dest_folder,
                                        "subject": subject,
                                        "from": sender,
                                        "date": str(getattr(msg, "date", ""))[:10],
                                    }
                                )
                            except Exception as err:
                                logger.warning(
                                    f"Failed to move email {uid} to {dest_folder}: {err}"
                                )

                    if actions_taken:
                        break
                except Exception as e:
                    logger.debug(f"Skipping search in {search_folder}: {e}")

            return {
                "status": "organized",
                "category": root_folder,
                "processed_count": processed,
                "organized_count": len(actions_taken),
                "folders_created": sorted(folders_created),
                "actions_taken": actions_taken,
            }

        # Mode B: Explicit rule-based processing
        with self._get_mailbox(folder=folder) as mailbox:
            messages = list(mailbox.fetch(AND(all=True), limit=limit, mark_seen=False))
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
                            try:
                                if not mailbox.folder.exists(dest):
                                    mailbox.folder.create(dest)
                            except Exception:
                                pass
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
            "organized_count": len(actions_taken),
            "actions_taken": actions_taken,
        }
