"""Mail specialist subagent enforcing autonomy policies and dispatching email tools."""

import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

from config import WorkOSConfig
from workos_engine.agents.base import BaseSubagent
from workos_engine.llm_client import extract_and_parse_json
from workos_engine.tools.mail_tools import MailToolKit
from workos_engine.types import AutonomyLevel, ExecutionResult, SubagentTask

logger = logging.getLogger(__name__)


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
        clean_args = dict(args)
        clean_args.pop("instruction", None)
        if "email_id" in clean_args and "uid" not in clean_args:
            clean_args["uid"] = clean_args.pop("email_id")
        elif "id" in clean_args and "uid" not in clean_args:
            clean_args["uid"] = clean_args.pop("id")
        if "folder_name" in clean_args and "folder" not in clean_args:
            clean_args["folder"] = clean_args.pop("folder_name")

        if t in ("search_emails", "search"):
            return self.toolkit.search_emails(**clean_args)
        elif t in ("fetch_email", "fetch", "read"):
            return self.toolkit.fetch_email(**clean_args)
        elif t in ("download_attachment", "download"):
            path = self.toolkit.download_attachment(**clean_args)
            return {"saved_path": path, "saved_paths": [path]}
        elif t in ("download_attachments", "download_all_attachments"):
            paths = self.toolkit.download_attachments(**clean_args)
            first_path = paths[0] if paths else ""
            return {"saved_paths": paths, "saved_path": first_path}
        elif t in ("create_draft", "draft"):
            return self.toolkit.create_draft(**clean_args)
        elif t in ("send_email", "send", "execute_send"):
            return self.execute_send(**clean_args)
        elif t in ("move_email", "move"):
            return self.toolkit.move_email(**clean_args)
        elif t in ("mark_email", "mark"):
            return self.toolkit.mark_email(**clean_args)
        elif t in ("list_folders", "folders"):
            return self.toolkit.list_folders()
        elif t in ("create_folder", "new_folder", "mkdir"):
            return self.toolkit.create_folder(**clean_args)
        elif t in ("organize_emails", "organize", "organize_folder"):
            return self.toolkit.organize_emails(**clean_args)
        else:
            raise ValueError(f"Unknown mail tool: {tool_name}")

    def _extract_organization_spec(
        self, goal: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Extracts structured email search parameters, temporal constraints, and classification criteria
        from the user's high-level goal using dynamic UTC system date grounding and LLM reasoning.
        """
        client = getattr(self, "model_client", None) or getattr(self, "llm_client", None)
        ctx = context or {}
        now_utc = datetime.now(UTC)
        current_date_str = now_utc.strftime("%Y-%m-%d")
        cognitive_context = ctx.get("cognitive_context") or ctx.get("memory") or ""

        default_folder = (
            self._resolve_destination_folder(ctx.get("folder_name") or ctx.get("category"), goal)
            or "Organized Mail"
        )
        spec: dict[str, Any] = {
            "target_folder": default_folder,
            "subfolder_rule": "none",
            "search_terms": [],
            "sender_patterns": [],
            "date_gte": None,
            "date_lt": None,
            "match_criteria": goal,
        }

        if not client or not hasattr(client, "generate"):
            return spec

        prompt = (
            f"You are the WorkOS Mail Organization Specialist.\n"
            f"Current Date: {current_date_str} (UTC).\n"
            f"User Goal: '{goal}'\n"
        )
        if cognitive_context:
            prompt += f"Context & Known Entities: {cognitive_context}\n"

        prompt += (
            f"\nExtract the email search and organization parameters into a clean JSON specification:\n"
            f"- target_folder: destination folder name without brackets\n"
            f"- subfolder_rule: attribute for subfolder naming if requested, otherwise 'none'\n"
            f"- search_terms: 2 to 4 topical search keywords from the goal (never use literal words 'subject' or 'body')\n"
            f"- sender_domains: list of top-level domains, email domains, or sender patterns implied by the goal\n"
            f"- date_gte: YYYY-MM-DD start date if specified or implied using current year {now_utc.year}, otherwise null\n"
            f"- date_lt: YYYY-MM-DD end date if specified or implied using current year {now_utc.year}, otherwise null\n"
            f"- match_criteria: concise description of what qualifies as an authentic match\n\n"
            f"Output strictly a JSON block like:\n"
            f"```json\n"
            f"{{\n"
            f'  "target_folder": "<folder>",\n'
            f'  "subfolder_rule": "<rule or none>",\n'
            f'  "search_terms": ["<term1>", "<term2>"],\n'
            f'  "sender_domains": ["<pattern1>", "<pattern2>"],\n'
            f'  "date_gte": "<YYYY-MM-DD or null>",\n'
            f'  "date_lt": "<YYYY-MM-DD or null>",\n'
            f'  "match_criteria": "<criteria>"\n'
            f"}}\n"
            f"```"
        )

        try:
            res = client.generate(prompt=prompt)
            parsed = extract_and_parse_json(res) if res else None
            if isinstance(parsed, dict):
                target_f = parsed.get("target_folder")
                if target_f and isinstance(target_f, str) and target_f.strip():
                    spec["target_folder"] = target_f.strip("[]'\" \t")
                if "subfolder_rule" in parsed and parsed["subfolder_rule"]:
                    spec["subfolder_rule"] = str(parsed["subfolder_rule"]).strip()
                if "search_terms" in parsed and isinstance(parsed["search_terms"], list):
                    meta_words = {"subject", "body", "keyword", "keywords", "search", "email", "emails", "mail", "mails"}
                    spec["search_terms"] = [
                        str(t).strip() for t in parsed["search_terms"]
                        if str(t).strip() and str(t).strip().lower() not in meta_words
                    ]
                sender_patterns = []
                for key in ("sender_domains", "sender_patterns", "domains", "senders"):
                    val = parsed.get(key)
                    if isinstance(val, list):
                        sender_patterns.extend([str(s).strip() for s in val if str(s).strip()])
                    elif isinstance(val, str) and val.strip():
                        sender_patterns.append(val.strip())
                if sender_patterns:
                    spec["sender_patterns"] = list(dict.fromkeys(sender_patterns))
                if parsed.get("date_gte"):
                    spec["date_gte"] = str(parsed["date_gte"]).strip()
                if parsed.get("date_lt"):
                    spec["date_lt"] = str(parsed["date_lt"]).strip()
                if parsed.get("match_criteria"):
                    spec["match_criteria"] = str(parsed["match_criteria"]).strip()
        except Exception as e:
            logger.debug(f"Spec extraction error: {e}")

        # Deterministic extraction of subfolder rule if user explicitly stated it
        if not spec.get("subfolder_rule") or spec["subfolder_rule"].lower() in ("none", "false", "no"):
            m_sub = re.search(
                r"sub\s*folders?\s+(?:based\s+upon|by|named)\s+([^,\n\.]+)", goal, re.IGNORECASE
            )
            if m_sub:
                spec["subfolder_rule"] = m_sub.group(1).strip()

        return spec

    def _generate_search_hypotheses(self, goal: str, context: dict[str, Any]) -> list[str]:
        spec = self._extract_organization_spec(goal, context)
        terms = spec.get("search_terms") or []
        senders = spec.get("sender_patterns") or []

        # Extract any specific known entities from cognitive context
        cognitive_context = context.get("cognitive_context") or context.get("memory") or ""
        context_entities = []
        if cognitive_context:
            acronyms = re.findall(r"\b[A-Z]{2,}\b", cognitive_context)
            context_entities.extend(acronyms)
            for m in re.findall(r"\(([^)]+)\)", cognitive_context):
                for part in re.split(r"[,;]+", m):
                    if len(part.strip()) >= 3:
                        context_entities.append(part.strip())

        combined = []
        for item in context_entities + senders + terms:
            clean = item.strip()
            if clean and clean.lower() not in [c.lower() for c in combined]:
                combined.append(clean)
        return combined[:8]

    def _retrieve_candidates(
        self,
        keywords: list[str],
        senders: list[str] | None = None,
        date_gte: Any | None = None,
        date_lt: Any | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if not keywords and not senders and not date_gte:
            return []
        try:
            return self.toolkit.search_emails_or(
                keywords=keywords,
                senders=senders,
                date_gte=date_gte,
                date_lt=date_lt,
                total_limit=limit,
            )
        except Exception as e:
            logger.warning(f"Failed candidate retrieval in search_emails_or: {e}")
            raise

    def _classify_with_llm(
        self,
        client: Any,
        goal: str,
        candidates: list[dict[str, Any]],
        cognitive_context: str = "",
        match_criteria: str = "",
        subfolder_rule: str = "",
        sender_patterns: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        all_matches: list[dict[str, Any]] = []
        all_rejected: list[dict[str, Any]] = []

        clean_context = ""
        if cognitive_context:
            seen_lines = set()
            unique_lines = []
            for line in cognitive_context.splitlines():
                line_str = line.strip()
                if line_str and line_str not in seen_lines:
                    seen_lines.add(line_str)
                    unique_lines.append(line_str)
            clean_context = "\n".join(unique_lines[:8])

        for cand in candidates:
            uid = str(cand.get("uid", ""))
            from_addr = (cand.get("from", "") or "").strip()
            subject = (cand.get("subject", "") or "").strip()

            raw_snippet = cand.get("snippet", "") or cand.get("text", "") or ""
            clean_snippet = re.sub(r"<[^>]+>", " ", raw_snippet)
            clean_snippet = re.sub(r"\s+", " ", clean_snippet).strip()

            candidate_payload = [{
                "uid": uid,
                "from": from_addr,
                "subject": subject,
                "date": cand.get("date", ""),
                "snippet": clean_snippet[:1200],
            }]

            prompt = f"You are the WorkOS Email Verification Specialist.\nUser Goal: '{goal}'\n"
            if match_criteria:
                prompt += f"Verification Criteria: {match_criteria}\n"
            if subfolder_rule:
                prompt += f"Subfolder Directive: {subfolder_rule}\n"
            if sender_patterns:
                clean_patterns = [str(p).lstrip("*") for p in sender_patterns if p]
                prompt += f"Target Sender Patterns / Domains: {clean_patterns}\n"
            if clean_context:
                prompt += f"Cognitive Context: {clean_context}\n"

            prompt += (
                "\nEvaluate this candidate email against the user goal and criteria.\n"
                "Task:\n"
                "1. is_match: true ONLY if this specific email is authentic communication matching the goal; false for spam, newsletters, third-party marketing, or irrelevant mail.\n"
                "2. subfolder: clean, canonical organization or institution name for the subfolder if matching, otherwise null.\n"
                "3. reason: brief explanation of why the email matches or does not match.\n\n"
                f"Candidate Email:\n{json.dumps(candidate_payload[0], indent=2)}\n\n"
                "Output strictly a JSON block like:\n"
                "```json\n"
                "{\n"
                '  "is_match": true,\n'
                '  "subfolder": "<clean organization or institution name>",\n'
                '  "reason": "<brief explanation>"\n'
                "}\n"
                "```"
            )

            try:
                res = client.generate(prompt=prompt)
                parsed = extract_and_parse_json(res) if res else None
                classifications = []
                if isinstance(parsed, dict):
                    if "classifications" in parsed and isinstance(parsed["classifications"], list):
                        classifications = [c for c in parsed["classifications"] if isinstance(c, dict)]
                    elif "is_match" in parsed or "subfolder" in parsed or "subcategory" in parsed:
                        classifications = [{"uid": uid, **parsed}]

                matched = False
                for item in classifications:
                    item_uid = str(item.get("uid", uid))
                    if item_uid == uid or item_uid in ("<uid>", "uid", "") or len(classifications) == 1:
                        is_match = (
                            item.get("is_match") is True or item.get("matches") is True
                        ) and item.get("is_direct_match") is not False
                        reason = str(item.get("reason", ""))
                        subfolder_val = str(item.get("subfolder") or item.get("subcategory") or "").strip()
                        subfolder_val = re.sub(r"^['\"]|['\"]$", "", subfolder_val).strip()

                        if is_match:
                            matched = True
                            if not subfolder_val or subfolder_val.lower() in ("none", "null"):
                                subfolder_val = "General"

                            all_matches.append({
                                "uid": uid,
                                "subcategory": subfolder_val,
                                "folder": cand.get("folder"),
                                "from": from_addr,
                                "subject": subject,
                                "date": cand.get("date", ""),
                            })
                        else:
                            all_rejected.append({"uid": uid, "reason": reason})
                        break

                if not matched and uid not in [m["uid"] for m in all_matches] and uid not in [r["uid"] for r in all_rejected]:
                    all_rejected.append({"uid": uid, "reason": "Unclassified candidate default"})
            except Exception as e:
                logger.warning(f"Classification error for candidate {uid}: {e}")
                all_rejected.append({"uid": uid, "reason": str(e)})

        return all_matches, all_rejected

    def _filter_and_classify_candidates(
        self,
        goal: str,
        candidates: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        match_criteria: str = "",
        subfolder_rule: str = "",
        sender_patterns: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not candidates:
            return [], []

        client = getattr(self, "model_client", None) or getattr(self, "llm_client", None)
        cognitive_context = ""
        if context:
            cognitive_context = context.get("cognitive_context") or context.get("memory") or ""

        if client and hasattr(client, "generate") and candidates:
            try:
                if not match_criteria:
                    spec = self._extract_organization_spec(goal, context)
                    match_criteria = spec.get("match_criteria", goal)
                    if not subfolder_rule:
                        subfolder_rule = spec.get("subfolder_rule", "by organization name")
                    if not sender_patterns:
                        sender_patterns = spec.get("sender_patterns")

                return self._classify_with_llm(
                    client=client,
                    goal=goal,
                    candidates=candidates,
                    cognitive_context=cognitive_context,
                    match_criteria=match_criteria,
                    subfolder_rule=subfolder_rule,
                    sender_patterns=sender_patterns,
                )
            except Exception as e:
                logger.warning(f"LLM candidate classification failed: {e}")

        # Conservative Safety Principle: In the absence of verified LLM classification,
        # never fabricate matches or guess subcategories heuristically.
        return [], [{"uid": str(c.get("uid", ""))} for c in candidates]

    def _resolve_destination_folder(self, requested_category: str | None, goal: str) -> str:
        try:
            folder_records = self.toolkit.list_folders()
            existing_folders = [f["name"] for f in folder_records]
        except Exception:
            existing_folders = []

        raw_name = (requested_category or "").strip()
        # Check if user specified explicit label/folder name in the prompt
        if not raw_name or raw_name == goal:
            m_quoted = re.search(
                r"(?:label|folder)\s+(?:called|named)\s+['\"]([^'\"]+)['\"]", goal, re.IGNORECASE
            )
            if m_quoted:
                raw_name = m_quoted.group(1).strip()
            else:
                m_unquoted = re.search(
                    r"(?:label|folder)\s+(?:called|named)\s+([^,\n\.]+?)(?:\s+(?:and\s+(?:then\s+)?sub|with\s+(?:a\s+)?sub|sub\s*(?:label|folder|category)|based\s+upon)|$)",
                    goal,
                    re.IGNORECASE,
                )
                if m_unquoted:
                    raw_name = m_unquoted.group(1).strip()
                else:
                    m_into = re.search(
                        r"(?:into|to|under)\s+(?:the\s+)?['\"]?([^'\",\n]+?)['\"]?\s+(?:label|folder)",
                        goal,
                        re.IGNORECASE,
                    )
                    if m_into:
                        raw_name = m_into.group(1).strip()
                    else:
                        raw_name = goal

        # Clean trailing connector clauses like "and then...", "with sub...", and strip brackets
        raw_name = re.sub(r"\s+and\s+(?:then\s+)?.*$", "", raw_name, flags=re.IGNORECASE).strip()
        raw_name = re.sub(r"\s+with\s+(?:a\s+)?sub.*$", "", raw_name, flags=re.IGNORECASE).strip()
        raw_name = raw_name.strip("[]'\" \t")

        # Normalize casing (e.g. "invoices and receipts" -> "Invoices And Receipts")
        clean_title = " ".join(word.capitalize() for word in raw_name.split()).strip("[]'\" \t")

        # Check existing folders for case-insensitive match
        for ef in existing_folders:
            if ef.lower() == raw_name.lower() or ef.lower() == clean_title.lower():
                return ef

        # Fast path: If clean_title is already a concise, well-formed category name, use it directly
        if clean_title and clean_title.lower() != goal.lower() and len(clean_title.split()) <= 4:
            return clean_title

        client = getattr(self, "model_client", None) or getattr(self, "llm_client", None)
        if not client or not hasattr(client, "generate"):
            return clean_title

        prompt = (
            f"Existing Mailbox Folders:\n{existing_folders}\n\n"
            f"User Goal / Requested Category: '{raw_name}'\n\n"
            f"Task:\n"
            f"1. Check if an existing folder in the mailbox already represents this category or intent. If so, return that exact existing folder name to avoid creating duplicate labels.\n"
            f"2. If no existing folder matches, fix any spelling typos or awkward casing in '{raw_name}' and return a clean, professional folder name without brackets.\n\n"
            f"Output strictly a JSON block like:\n"
            f"```json\n"
            f'{{"target_folder": "<clean folder name>"}}\n'
            f"```"
        )
        try:
            res = client.generate(prompt=prompt)
            if res:
                parsed = extract_and_parse_json(res)
                if isinstance(parsed, dict):
                    target = parsed.get("target_folder")
                    if target and isinstance(target, str) and target.strip():
                        return target.strip("[]'\" \t")
        except Exception as e:
            logger.debug(f"Folder resolution error: {e}")

        return clean_title

    def autonomous_organize(self, goal: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        First-principles autonomous email organization pipeline:
        1. Spec Extraction (grounded by UTC runtime date)
        2. Candidate Retrieval (bounded by dates, high recall)
        3. Precision Per-Email Filtering
        4. Folder Taxonomy Resolution
        5. Batch Atomic Execution & Receipt Generation
        """
        # Phase 1: Spec Extraction
        spec = self._extract_organization_spec(goal, context)
        raw_dest = spec.get("target_folder") or "Organized Mail"
        req_cat = (
            context.get("category")
            or context.get("folder_name")
            or context.get("label")
            or raw_dest
        )
        dest_folder = self._resolve_destination_folder(req_cat, goal) or raw_dest

        # Phase 2: Candidate Retrieval
        search_terms = spec.get("search_terms") or []
        sender_patterns = spec.get("sender_patterns") or []
        date_gte = spec.get("date_gte")
        date_lt = spec.get("date_lt")

        candidates = self._retrieve_candidates(
            keywords=search_terms,
            senders=sender_patterns,
            date_gte=date_gte,
            date_lt=date_lt,
            limit=500,
        )

        # Phase 3: Precision Filtering
        matches, rejected = self._filter_and_classify_candidates(
            goal=goal,
            candidates=candidates,
            context=context,
            match_criteria=spec.get("match_criteria", goal),
            subfolder_rule=spec.get("subfolder_rule", "by entity name"),
            sender_patterns=sender_patterns,
        )

        if not matches:
            receipt = (
                f"Scanned {len(candidates)} candidate emails matching search criteria. "
                f"Zero emails verified as matching '{goal}'. No folders created or emails moved."
            )
            return {
                "status": "completed",
                "category": dest_folder,
                "candidates_scanned": len(candidates),
                "organized_count": 0,
                "actions_taken": [],
                "rejected_count": len(rejected),
                "message": receipt,
                "receipt": receipt,
            }

        # Phase 5: High-speed Batch Execution
        actions_taken = []
        uids_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
        subfolder_rule = (spec.get("subfolder_rule") or "").strip().lower()
        has_subfolders = subfolder_rule not in ("none", "false", "no", "", "null")
        for m in matches:
            subcat = (
                (m.get("subcategory") or "").strip().replace("/", "-").replace("\\", "-")
                if has_subfolders
                else ""
            )
            final_folder = f"{dest_folder}/{subcat}" if subcat else dest_folder
            source_f = m.get("folder") or "INBOX"
            uids_by_group.setdefault((final_folder, source_f), []).append(m)

        for (final_folder, source_f), group_matches in uids_by_group.items():
            self.toolkit.create_folder(final_folder)
            uids = [str(m["uid"]) for m in group_matches if m.get("uid")]
            if not uids:
                continue

            moved_successfully = False
            try:
                with self.toolkit._get_mailbox(folder=source_f) as mb:
                    mb.move(uids, final_folder)
                moved_successfully = True
            except Exception as e:
                logger.debug(f"Batch move failed in {source_f} ({e}), trying All Mail fallback...")
                is_gmail = "gmail" in (self.config.imap_host or "").lower()
                if is_gmail and source_f != "[Gmail]/All Mail":
                    try:
                        with self.toolkit._get_mailbox(folder="[Gmail]/All Mail") as mb:
                            mb.move(uids, final_folder)
                        moved_successfully = True
                    except Exception as fallback_err:
                        logger.debug(
                            f"Fallback batch move failed for {final_folder}: {fallback_err}"
                        )

            if moved_successfully:
                for m in group_matches:
                    actions_taken.append(
                        {
                            "uid": str(m.get("uid", "")),
                            "action": "move",
                            "destination": final_folder,
                            "subcategory": (m.get("subcategory") or "").strip(),
                            "from": m.get("from", ""),
                            "subject": m.get("subject", ""),
                            "date": m.get("date", ""),
                        }
                    )
            else:
                for m in group_matches:
                    uid = str(m.get("uid", ""))
                    try:
                        self.toolkit.move_email(
                            uid=uid, destination_folder=final_folder, source_folder=source_f
                        )
                        actions_taken.append(
                            {
                                "uid": uid,
                                "action": "move",
                                "destination": final_folder,
                                "subcategory": (m.get("subcategory") or "").strip(),
                                "from": m.get("from", ""),
                                "subject": m.get("subject", ""),
                                "date": m.get("date", ""),
                            }
                        )
                    except Exception as err:
                        logger.warning(
                            f"Failed to move verified email {uid} to {final_folder}: {err}"
                        )

        subcat_count = len({m.get("subcategory") for m in matches if m.get("subcategory")})
        receipt = (
            f"Organized {len(actions_taken)} verified emails into '{dest_folder}' across "
            f"{subcat_count} subfolders. Rejected {len(rejected)} noise emails."
        )
        return {
            "status": "organized",
            "category": dest_folder,
            "candidates_scanned": len(candidates),
            "organized_count": len(actions_taken),
            "actions_taken": actions_taken,
            "rejected_count": len(rejected),
            "receipt": receipt,
            "message": receipt,
        }

    def execute(self, task: SubagentTask) -> ExecutionResult:
        """Executes a delegated email task instruction or runs autonomous micro-ReAct loop."""
        instruction = (task.instruction or "").lower().strip()
        ctx = dict(task.context or {})

        # Autonomous organization pipeline
        combined_text = f"{instruction} {ctx.get('goal', '')} {ctx.get('step_description', '')} {ctx.get('query', '')}".lower()
        if instruction in ("organize", "organize_emails") or any(
            w in combined_text
            for w in ("organize", "sort", "subfolder", "subfolders", "move_emails")
        ):
            goal = (
                ctx.get("goal")
                or ctx.get("step_description")
                or ctx.get("query")
                or task.instruction
            )
            try:
                data = self.autonomous_organize(goal=goal, context=ctx)
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=True,
                    data=data,
                    artifacts=[],
                )
            except Exception as e:
                logger.exception(f"Error in autonomous_organize: {e}")
                return ExecutionResult(
                    task_id=task.task_id,
                    agent_name=self.name,
                    success=False,
                    error=str(e),
                )

        # Direct mapped tool calls
        mapped_tools = {
            "search_emails": "search_emails",
            "search": "search_emails",
            "fetch_email": "fetch_email",
            "fetch": "fetch_email",
            "read": "fetch_email",
            "download_attachment": "download_attachment",
            "download_attachments": "download_attachments",
            "download_all_attachments": "download_attachments",
            "download": "download_attachments",
            "create_draft": "create_draft",
            "draft": "create_draft",
            "send_email": "send_email",
            "send": "send_email",
            "execute_send": "send_email",
            "move_email": "move_email",
            "move": "move_email",
            "mark_email": "mark_email",
            "mark": "mark_email",
            "list_folders": "list_folders",
            "create_folder": "create_folder",
            "create_label": "create_folder",
            "add_label": "create_folder",
            "new_label": "create_folder",
            "label": "create_folder",
            "labels": "list_folders",
            "list_labels": "list_folders",
            "organize_emails": "organize_emails",
            "organize": "organize_emails",
            "create_subfolder": "organize_emails",
            "create_subfolders": "organize_emails",
            "move_emails_to_folder": "organize_emails",
            "move_to_folder": "organize_emails",
        }

        if instruction in mapped_tools:
            tool_name = mapped_tools[instruction]
            try:
                artifacts: list[str] = []
                data = self.execute_tool(tool_name, ctx)
                if isinstance(data, dict):
                    if "saved_path" in data and data["saved_path"] and os.path.exists(str(data["saved_path"])):
                        artifacts.append(str(data["saved_path"]))
                    if "saved_paths" in data and isinstance(data["saved_paths"], list):
                        for p in data["saved_paths"]:
                            if isinstance(p, str) and os.path.exists(p) and p not in artifacts:
                                artifacts.append(p)
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
                "name": "download_attachments",
                "description": "Download all attachments from an email by UID to a local directory, returning a list of saved file paths.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {
                            "type": "string",
                            "description": "Unique identifier of the email containing attachments.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "IMAP folder name. Defaults to 'INBOX'.",
                        },
                        "save_dir": {
                            "type": "string",
                            "description": "Local destination directory. Defaults to './downloads'.",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Optional regex pattern to filter attachment filenames.",
                        },
                    },
                    "required": ["uid"],
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
                "description": "Automatically categorize and move emails into folders or subfolders based on category or custom rules.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Destination category folder name (e.g. 'Archive', 'Receipts', 'Invoices', 'Projects').",
                        },
                        "sender": {
                            "type": "string",
                            "description": "Sender email or name to filter.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Subject keywords to filter.",
                        },
                        "text": {
                            "type": "string",
                            "description": "Body text or search keywords to filter.",
                        },
                        "date_gte": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD) to filter emails.",
                        },
                        "date_lt": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD) to filter emails.",
                        },
                        "seen": {
                            "type": "boolean",
                            "description": "Filter by read (true) or unread (false) emails.",
                        },
                        "rules": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Optional custom rule dictionaries with match criteria and destination.",
                        },
                        "folder": {
                            "type": "string",
                            "description": "IMAP source folder. Defaults to 'INBOX'.",
                        },
                    },
                },
            },
            {
                "name": "list_folders",
                "description": "List all existing IMAP folders / mailboxes and their hierarchy delimiter.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "create_folder",
                "description": "Create a new IMAP folder or nested subfolder (e.g. 'Archive/2026', 'Projects/Alpha').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "folder": {
                            "type": "string",
                            "description": "Folder or subfolder name to create. Use '/' for nested folders.",
                        },
                    },
                    "required": ["folder"],
                },
            },
        ]
