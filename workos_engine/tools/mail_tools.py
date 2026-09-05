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

    def resolve_folder_name(self, folder: str | None, mailbox: MailBox | None = None) -> str:
        """
        Resolves canonical or logical mailbox aliases (e.g. 'Drafts', 'Sent', 'Trash', 'Spam', 'All Mail')
        to the server's actual folder name, respecting Gmail namespace conventions and IMAP special-use flags.
        """
        if not folder:
            return "INBOX"

        f_clean = folder.strip()
        f_lower = f_clean.lower()
        is_gmail = "gmail" in (self.config.imap_host or "").lower()

        if is_gmail:
            if f_lower in ("drafts", "draft", "[gmail]/drafts"):
                return "[Gmail]/Drafts"
            if f_lower in ("sent", "sent mail", "sent messages", "sent items", "[gmail]/sent mail"):
                return "[Gmail]/Sent Mail"
            if f_lower in ("trash", "bin", "[gmail]/trash", "[gmail]/bin"):
                return "[Gmail]/Bin"
            if f_lower in ("spam", "junk", "[gmail]/spam"):
                return "[Gmail]/Spam"
            if f_lower in ("all", "all mail", "[gmail]/all mail"):
                return "[Gmail]/All Mail"
            if f_lower in ("starred", "[gmail]/starred"):
                return "[Gmail]/Starred"
            if f_lower in ("important", "[gmail]/important"):
                return "[Gmail]/Important"
            if f_lower == "inbox":
                return "INBOX"

        if mailbox is not None:
            discovered = self._discover_folder(mailbox, f_clean)
            if discovered:
                return discovered

        return f_clean

    def _discover_folder(self, mailbox: MailBox, folder: str) -> str | None:
        """Discovers folder from server list matching by name, special-use flags, or hierarchy."""
        f_clean = folder.strip().lower()
        flag_map = {
            "drafts": "\\Drafts",
            "draft": "\\Drafts",
            "sent": "\\Sent",
            "sent mail": "\\Sent",
            "sent items": "\\Sent",
            "trash": "\\Trash",
            "bin": "\\Trash",
            "spam": "\\Junk",
            "junk": "\\Junk",
            "all mail": "\\All",
            "all": "\\All",
            "archive": "\\Archive",
            "starred": "\\Flagged",
            "flagged": "\\Flagged",
        }
        target_flag = flag_map.get(f_clean)
        try:
            folders = list(mailbox.folder.list())
            if target_flag:
                for f in folders:
                    flags = [flag.lower() for flag in (getattr(f, "flags", ()) or ())]
                    if target_flag.lower() in flags:
                        return f.name
            for f in folders:
                if f.name.lower() == f_clean:
                    return f.name
            for f in folders:
                if (
                    f.name.lower().endswith(f_clean)
                    or f.name.lower().endswith(f"/{f_clean}")
                    or f.name.lower().endswith(f".{f_clean}")
                ):
                    return f.name
        except Exception as e:
            logger.debug(f"Folder discovery failed: {e}")
        return None

    def _get_mailbox(self, folder: str = "INBOX", timeout: float = 30.0) -> MailBox:
        """Connects and logs into the IMAP mailbox on the given folder with dynamic fallback."""
        if not self.config.imap_host:
            raise ValueError("IMAP host is not configured.")
        target_folder = self.resolve_folder_name(folder)
        mb = MailBox(self.config.imap_host, port=self.config.imap_port, timeout=timeout)
        try:
            return mb.login(
                self.config.imap_user, self.config.imap_password, initial_folder=target_folder
            )
        except Exception as e:
            err_str = str(e).lower()
            if "nonexistent" in err_str or "unknown mailbox" in err_str or "failure" in err_str:
                mb_fallback = MailBox(self.config.imap_host, port=self.config.imap_port, timeout=timeout)
                client = mb_fallback.login(
                    self.config.imap_user, self.config.imap_password, initial_folder="INBOX"
                )
                resolved = self._discover_folder(client, folder)
                if resolved and resolved != "INBOX":
                    try:
                        client.folder.set(resolved)
                        return client
                    except Exception:
                        pass
                if folder.upper() == "INBOX":
                    raise
                return client
            raise

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

        def _format_msg_dict(msg: Any, current_folder: str = "INBOX") -> dict[str, Any]:
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
                "folder": current_folder,
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
                "attachment_names": [
                    getattr(a, "filename", "") for a in attachments if getattr(a, "filename", "")
                ],
                "snippet": body_snippet,
            }

        results: list[dict[str, Any]] = []
        seen_uids: set[str] = set()

        resolved_folder = self.resolve_folder_name(folder)
        folders_to_search = [resolved_folder]
        is_gmail = "gmail" in (self.config.imap_host or "").lower()
        if is_gmail and folder.upper() == "INBOX":
            folders_to_search.append("[Gmail]/All Mail")

        for search_folder in folders_to_search:
            try:
                with self._get_mailbox(folder=search_folder) as mailbox:
                    imap_query = AND(**criteria) if criteria else AND(all=True)
                    messages = list(
                        mailbox.fetch(
                            imap_query, limit=limit, reverse=True, mark_seen=False, bulk=50
                        )
                    )

                    # If exact phrase match returned no results, only fall back if ALL significant terms are present
                    if not messages and search_text:
                        terms = [
                            w.strip().lower()
                            for w in re.split(r"[\s,]+", search_text)
                            if len(w.strip()) >= 3
                        ]
                        if len(terms) > 1:
                            primary_term = max(terms, key=len)
                            term_criteria = dict(criteria)
                            term_criteria["text"] = primary_term
                            for msg in mailbox.fetch(
                                AND(**term_criteria),
                                limit=limit,
                                reverse=True,
                                mark_seen=False,
                                bulk=50,
                            ):
                                uid = getattr(msg, "uid", "")
                                body_content = (
                                    getattr(msg, "text", "") or getattr(msg, "html", "") or ""
                                ).lower()
                                subj_content = (getattr(msg, "subject", "") or "").lower()
                                full_content = f"{subj_content} {body_content}"
                                # Require ALL terms to be present
                                if all(t in full_content for t in terms):
                                    if uid and uid not in seen_uids:
                                        seen_uids.add(uid)
                                        messages.append(msg)

                    for msg in messages:
                        uid = getattr(msg, "uid", "")
                        if uid and uid in seen_uids:
                            continue
                        if uid:
                            seen_uids.add(uid)
                        results.append(_format_msg_dict(msg, current_folder=search_folder))
                        if len(results) >= limit:
                            break

                if len(results) >= limit:
                    break
            except Exception as e:
                err_str = str(e).lower()
                if (
                    "auth" in err_str
                    or "login" in err_str
                    or "credential" in err_str
                    or "connection refused" in err_str
                ):
                    raise
                if "nonexistent" in err_str or "unknown mailbox" in err_str:
                    logger.info(f"Mailbox '{search_folder}' does not exist on server; skipping.")
                else:
                    logger.warning(f"IMAP search error in {search_folder}: {e}")

        return results

    def search_emails_or(
        self,
        keywords: list[str] | None = None,
        senders: list[str] | None = None,
        date_gte: Any | None = None,
        date_lt: Any | None = None,
        folder: str = "INBOX",
        limit_per_keyword: int = 150,
        total_limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        High-recall disjunctive search fetching emails matching ANY of the provided keywords or senders,
        optionally bounded by temporal date filters.
        Reuses IMAP connection across search folders for high-speed retrieval.
        """
        raw_keywords = list(keywords or [])
        raw_senders = list(senders or [])
        if not raw_keywords and not raw_senders and not date_gte and not date_lt:
            return []

        def _clean_ascii(s: str) -> str:
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

        parsed_gte = self._parse_date(date_gte) if date_gte else None
        parsed_lt = self._parse_date(date_lt) if date_lt else None
        base_criteria: dict[str, Any] = {}
        if parsed_gte:
            base_criteria["date_gte"] = parsed_gte
        if parsed_lt:
            base_criteria["date_lt"] = parsed_lt

        resolved_folder = self.resolve_folder_name(folder)
        folders_to_search = [resolved_folder]
        is_gmail = "gmail" in (self.config.imap_host or "").lower()
        if is_gmail and folder.upper() in ("INBOX", "ALL", "ALL MAIL", "[GMAIL]/ALL MAIL"):
            folders_to_search = ["[Gmail]/All Mail"]

        results_by_uid: dict[str, dict[str, Any]] = {}

        # Build list of criteria
        criteria_list: list[dict[str, Any]] = []
        for s in raw_senders:
            s_clean = _clean_ascii(s.strip()).lstrip("* \t")
            if s_clean and s_clean not in ("*", "@", "*@*"):
                criteria_list.append({"from_": s_clean, **base_criteria})

        for kw in raw_keywords:
            kw_clean = _clean_ascii(kw.strip())
            if not kw_clean:
                continue
            if kw_clean.startswith(".") or "@" in kw_clean:
                criteria_list.append({"from_": kw_clean, **base_criteria})
            elif "." in kw_clean and " " not in kw_clean:
                criteria_list.append({"from_": kw_clean, **base_criteria})
            else:
                criteria_list.append({"subject": kw_clean, **base_criteria})

        if not criteria_list and base_criteria:
            criteria_list.append(dict(base_criteria))

        for search_folder in folders_to_search:
            try:
                with self._get_mailbox(folder=search_folder) as mailbox:
                    for crit in criteria_list:
                        try:
                            for msg in mailbox.fetch(
                                AND(**crit),
                                limit=limit_per_keyword,
                                reverse=True,
                                mark_seen=False,
                                bulk=50,
                            ):
                                uid = getattr(msg, "uid", "")
                                if uid and uid not in results_by_uid:
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

                                    results_by_uid[uid] = {
                                        "uid": uid,
                                        "folder": search_folder,
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
                                    if len(results_by_uid) >= total_limit:
                                        break
                        except Exception as e:
                            logger.debug(
                                f"Search fetch error for crit {crit} in {search_folder}: {e}"
                            )

                        if len(results_by_uid) >= total_limit:
                            break
            except Exception as folder_err:
                logger.debug(f"Folder search error in {search_folder}: {folder_err}")

            if len(results_by_uid) >= total_limit:
                break

        return list(results_by_uid.values())[:total_limit]

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

    def download_attachments(
        self,
        uid: str,
        folder: str = "INBOX",
        save_dir: str = "./downloads",
        pattern: str | None = None,
    ) -> list[str]:
        """Download all attachments from an email by UID to a local directory, returning saved file paths."""
        os.makedirs(save_dir, exist_ok=True)
        saved_paths: list[str] = []
        with self._get_mailbox(folder=folder) as mailbox:
            messages = list(mailbox.fetch(AND(uid=uid), mark_seen=False))
            if not messages:
                raise ValueError(f"Email with UID '{uid}' not found in folder '{folder}'.")
            msg = messages[0]

            for att in getattr(msg, "attachments", []) or []:
                fname = getattr(att, "filename", "") or ""
                if not fname:
                    continue
                if pattern and not re.search(pattern, fname, re.IGNORECASE):
                    continue
                saved_path = os.path.join(save_dir, fname)
                with open(saved_path, "wb") as f:
                    f.write(getattr(att, "payload", b""))
                saved_paths.append(os.path.abspath(saved_path))

        return saved_paths

    def download_attachment(
        self,
        uid: str,
        attachment_filename: str | None = None,
        folder: str = "INBOX",
        save_dir: str = "./downloads",
    ) -> str:
        """Download an attachment from an email by UID to a local directory."""
        if not attachment_filename or attachment_filename.lower() in ("all", "*"):
            paths = self.download_attachments(uid=uid, folder=folder, save_dir=save_dir)
            if not paths:
                raise FileNotFoundError(f"No attachments found in email UID '{uid}'.")
            return paths[0]

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

            return os.path.abspath(saved_path)

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
            target_folder = self.resolve_folder_name(folder)
            with self._get_mailbox(folder=target_folder) as mailbox:
                mailbox.append(msg.as_bytes(), folder=target_folder)
        except Exception as e:
            # Non-blocking if IMAP server is not reachable or drafts folder doesn't exist
            logger.debug(f"Failed to append draft to {folder}: {e}")

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
        try:
            with self._get_mailbox(folder=source_folder) as mailbox:
                mailbox.move(uid, destination_folder)
        except Exception:
            is_gmail = "gmail" in (self.config.imap_host or "").lower()
            if is_gmail and source_folder == "INBOX":
                with self._get_mailbox(folder="[Gmail]/All Mail") as mailbox:
                    mailbox.move(uid, destination_folder)
            else:
                raise
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
        seen: bool | None = None,
        sender: str | None = None,
        subject: str | None = None,
        text: str | None = None,
        folder: str = "INBOX",
        limit: int = 500,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Organize emails in the mailbox.
        Supports:
        1. Rule-based criteria: rules=[{"match": {...}, "action": "move", "destination": "..."}]
        2. Category-based organization: moves all emails matching search criteria to the specified category folder.
        """
        actions_taken = []
        processed = 0

        # Mode A: Rule-based processing
        if rules:
            with self._get_mailbox(folder=folder) as mailbox:
                messages = list(mailbox.fetch(AND(all=True), limit=limit, mark_seen=False, bulk=50))
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

                        if matched and "text" in match_criteria:
                            text_val = match_criteria["text"].lower()
                            msg_text = (
                                getattr(msg, "text", "") or getattr(msg, "html", "") or ""
                            ).lower()
                            if text_val not in msg_text:
                                matched = False

                        if matched:
                            action = rule.get("action", "move")
                            uid = getattr(msg, "uid", "")
                            msg_subj = getattr(msg, "subject", "")

                            if action == "move":
                                dest = rule.get("destination", "Archive")
                                self.create_folder(dest)
                                mailbox.move(uid, dest)
                                actions_taken.append(
                                    {
                                        "uid": uid,
                                        "action": "move",
                                        "destination": dest,
                                        "subject": msg_subj,
                                    }
                                )
                            elif action == "mark_seen":
                                mailbox.flag(uid, ["\\Seen"], True)
                                actions_taken.append(
                                    {
                                        "uid": uid,
                                        "action": "mark_seen",
                                        "subject": msg_subj,
                                    }
                                )
                            elif action == "mark_unseen":
                                mailbox.flag(uid, ["\\Seen"], False)
                                actions_taken.append(
                                    {
                                        "uid": uid,
                                        "action": "mark_unseen",
                                        "subject": msg_subj,
                                    }
                                )
                            elif action == "mark_flagged":
                                mailbox.flag(uid, ["\\Flagged"], True)
                                actions_taken.append(
                                    {
                                        "uid": uid,
                                        "action": "mark_flagged",
                                        "subject": msg_subj,
                                    }
                                )
                            break  # Apply first matching rule per email

            return {
                "status": "organized",
                "processed_count": processed,
                "organized_count": len(actions_taken),
                "actions_taken": actions_taken,
            }

        # Mode B: Direct category folder organization
        dest_folder = category or kwargs.get("destination") or kwargs.get("folder_name")
        if not dest_folder:
            return {
                "status": "skipped",
                "category": None,
                "organized_count": 0,
                "message": "Organization skipped: No category or destination folder provided.",
            }

        effective_sender = sender or kwargs.get("domain_pattern") or kwargs.get("from_")
        effective_text = text or kwargs.get("query")

        self.create_folder(dest_folder)

        matched_emails = self.search_emails(
            sender=effective_sender,
            subject=subject,
            date_gte=date_gte,
            date_lt=date_lt,
            seen=seen,
            text=effective_text,
            folder=folder,
            limit=limit,
        )

        moved_uids = set()
        for email_info in matched_emails:
            uid = str(email_info.get("uid", ""))
            if not uid or uid in moved_uids:
                continue

            try:
                self.move_email(uid=uid, destination_folder=dest_folder, source_folder=folder)
                moved_uids.add(uid)
                actions_taken.append(
                    {
                        "uid": uid,
                        "action": "move",
                        "destination": dest_folder,
                        "subject": email_info.get("subject", ""),
                        "from": email_info.get("from", ""),
                        "date": email_info.get("date", ""),
                    }
                )
            except Exception as err:
                logger.warning(f"Failed to move email {uid} to {dest_folder}: {err}")

        return {
            "status": "organized",
            "category": dest_folder,
            "processed_count": len(matched_emails),
            "organized_count": len(actions_taken),
            "actions_taken": actions_taken,
        }
