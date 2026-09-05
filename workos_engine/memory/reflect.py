import json
import logging
import re
import uuid
from typing import Any

from workos_engine.memory.db import MemoryDB
from workos_engine.memory.loci import SpatialLociManager
from workos_engine.types import MemoryItem, MemoryNetwork

logger = logging.getLogger(__name__)


class MemoryReflector:
    """
    Cognitive reflection module that processes conversation events and task outputs,
    extracting facts, experiences, entities, and beliefs/preferences into memory.
    """

    def __init__(self, loci_manager: SpatialLociManager | None = None):
        self.loci = loci_manager or SpatialLociManager()

    def reflect_and_update(
        self,
        db: MemoryDB,
        conversation_events: list[Any],
        model_client: Any | None = None,
    ) -> list[str]:
        """
        Consolidates conversation events into structured memory items.
        Uses LLM client if available; falls back to robust heuristic rule extraction.
        """
        if not conversation_events:
            return []

        extracted_items: list[MemoryItem] = []

        if model_client is not None:
            extracted_items = self._reflect_with_model(conversation_events, model_client)
        else:
            extracted_items = self._reflect_heuristically(conversation_events)

        created_ids: list[str] = []
        for item in extracted_items:
            # If belief, supersede existing active beliefs under same key
            if item.network == MemoryNetwork.BELIEFS:
                db.supersede_key(
                    key=item.key,
                    wing=item.wing,
                    hall=item.hall,
                    new_id=item.id,
                )

            db.insert_memory(item)

            if item.network == MemoryNetwork.ENTITIES:
                db.upsert_entity(
                    entity_name=item.key,
                    entity_type=item.hall,
                    summary=item.content,
                    attributes=item.metadata,
                )

            created_ids.append(item.id)

        return created_ids

    def _reflect_heuristically(self, conversation_events: list[Any]) -> list[MemoryItem]:
        """
        Rule-based pattern matching to extract preferences, facts, and task experiences.
        """
        items: list[MemoryItem] = []

        for event in conversation_events:
            text = ""
            role = "user"
            if isinstance(event, dict):
                text = event.get("content", "")
                role = event.get("role", "user")
            elif isinstance(event, str):
                text = event
            elif hasattr(event, "content"):
                text = getattr(event, "content", "")
                role = getattr(event, "role", "user")

            if not text:
                continue

            # 1. Beliefs / Preferences extraction
            belief_match = re.search(
                r"(?:always|prefer|prefers|format your|summary format|please use|style:)\s*([^\.\n]+)",
                text,
                re.IGNORECASE,
            )
            if belief_match and role in ["user", "system"]:
                pref_text = text.strip()
                wing, hall = "workflows", "preferences"
                raw_slug_words = re.findall(r"[a-zA-Z0-9]+", pref_text.lower())
                stop_words = {"always", "prefer", "prefers", "please", "use", "my", "the", "a", "an", "is", "in", "to", "for", "your", "our"}
                content_words = [w for w in raw_slug_words if w not in stop_words]
                slug = "_".join(content_words[:3]) if content_words else "general"
                key = f"pref_{slug}"
                items.append(
                    MemoryItem(
                        id=f"mem_{uuid.uuid4().hex[:12]}",
                        network=MemoryNetwork.BELIEFS,
                        wing=wing,
                        hall=hall,
                        key=key,
                        content=pref_text,
                        confidence=0.95,
                        metadata={"source": "conversation_reflection"},
                    )
                )

            # 2. Fact extraction (e.g. "Fact:", "The server port is...", "X is Y")
            fact_match = re.search(
                r"(?:fact:\s*|note:\s*|the\s+([a-zA-Z0-9_\s]+)\s+is\s+([^\.\n]+))",
                text,
                re.IGNORECASE,
            )
            if fact_match:
                content = text.replace("Fact:", "").replace("fact:", "").strip()
                wing, hall = self.loci.route_topic(content, "knowledge", "general")
                # Generate clean key
                words = re.findall(r"\w+", content.lower())
                key = "_".join(words[:3]) if words else "fact_item"
                items.append(
                    MemoryItem(
                        id=f"mem_{uuid.uuid4().hex[:12]}",
                        network=MemoryNetwork.FACTS,
                        wing=wing,
                        hall=hall,
                        key=key,
                        content=content,
                        confidence=0.9,
                        metadata={"extracted": True},
                    )
                )

            # 3. Experience extraction (e.g. "Processed invoice #101", task completion)
            if any(
                k in text.lower() for k in ["completed", "processed", "executed", "task result"]
            ):
                wing, hall = "workflows", "runs"
                words = re.findall(r"\w+", text.lower())
                key = "_".join(words[:3]) if words else "task_run"
                items.append(
                    MemoryItem(
                        id=f"mem_{uuid.uuid4().hex[:12]}",
                        network=MemoryNetwork.EXPERIENCES,
                        wing=wing,
                        hall=hall,
                        key=key,
                        content=text.strip(),
                        confidence=1.0,
                        metadata={"type": "task_log"},
                    )
                )

            # 4. User profile / domain context extraction
            profile_match = re.search(
                r"(?:my name is|i am applying to|i am working on|i live in|my email is|call me)\s+([^\.\n]+)",
                text,
                re.IGNORECASE,
            )
            if profile_match and role in ["user", "system"]:
                content = text.strip()
                wing, hall = "people", "contacts"
                sub_words = re.findall(r"\w+", profile_match.group(1).lower())
                key = "user_profile_" + ("_".join(sub_words[:2]) if sub_words else "info")
                items.append(
                    MemoryItem(
                        id=f"mem_{uuid.uuid4().hex[:12]}",
                        network=MemoryNetwork.FACTS,
                        wing=wing,
                        hall=hall,
                        key=key,
                        content=content,
                        confidence=0.9,
                        metadata={"source": "user_dialogue"},
                    )
                )

        return items

    # Compatibility alias
    _reflect_rule_based = _reflect_heuristically

    def _reflect_with_model(
        self, conversation_events: list[Any], model_client: Any
    ) -> list[MemoryItem]:
        """
        Uses local Ollama LLM to extract structured memories across the 4 Hindsight networks.
        """
        from workos_engine.llm_client import extract_and_parse_json

        events_json = json.dumps(
            [e if isinstance(e, (dict, str)) else str(e) for e in conversation_events]
        )
        prompt = f"""You are the WorkOS Cognitive Memory Reflector.
Analyze the following conversation events and extract structured memory items to retain.
Categorize each item into one of the 4 Hindsight networks:
- 'facts': Permanent objective knowledge (system configs, world facts, credentials, specs)
- 'experiences': Episodic execution logs, task results, and actions taken
- 'entities': Named entities (people, tools, institutions, universities, projects)
- 'beliefs': Working assumptions, user preferences, mental models, formatting rules

Spatial Loci organization:
- wing: 'people', 'projects', 'workflows', 'knowledge', or 'general'
- hall: specific topic corridor (e.g., 'contacts', 'preferences', 'decisions', 'runs', 'admissions')

Events to analyze:
{events_json}

Return strictly a JSON array of memory objects in this format:
[
  {{
    "network": "facts|experiences|entities|beliefs",
    "wing": "workflows",
    "hall": "preferences",
    "key": "unique_semantic_key",
    "content": "clear, concise declarative statement",
    "confidence": 1.0,
    "metadata": {{}}
  }}
]
"""
        resp_text = ""
        try:
            if hasattr(model_client, "generate"):
                resp_text = model_client.generate(prompt=prompt, format="json").strip()
            elif hasattr(model_client, "chat"):
                resp_text = model_client.chat(
                    [{"role": "user", "content": prompt}], format="json"
                ).strip()
            else:
                return self._reflect_heuristically(conversation_events)
        except Exception as e:
            logger.warning(f"Model reflection call failed, falling back to heuristics: {e}")
        parsed = extract_and_parse_json(resp_text)
        if not parsed:
            return self._reflect_heuristically(conversation_events)

        if isinstance(parsed, dict):
            parsed_list = (
                parsed.get("memories") or parsed.get("items") or parsed.get("data") or [parsed]
            )
        elif isinstance(parsed, list):
            parsed_list = parsed
        else:
            parsed_list = []

        items: list[MemoryItem] = []
        for p in parsed_list:
            if not isinstance(p, dict) or "network" not in p:
                continue
            try:
                network_val = str(p["network"]).lower().strip()
                raw_content = p.get("content", "")
                content_str = (
                    json.dumps(raw_content)
                    if isinstance(raw_content, (dict, list))
                    else str(raw_content).strip()
                )
                if not content_str:
                    continue

                raw_key = p.get("key", "item")
                key_str = "_".join(re.findall(r"\w+", str(raw_key).lower())) or "item"
                raw_conf = p.get("confidence", 1.0)
                try:
                    conf_val = float(raw_conf)
                except (ValueError, TypeError):
                    conf_val = 1.0

                raw_meta = p.get("metadata", {})
                meta_val = raw_meta if isinstance(raw_meta, dict) else {}

                items.append(
                    MemoryItem(
                        id=f"mem_{uuid.uuid4().hex[:12]}",
                        network=MemoryNetwork(network_val),
                        wing=str(p.get("wing", "general")).strip().lower(),
                        hall=str(p.get("hall", "general")).strip().lower(),
                        key=key_str,
                        content=content_str,
                        confidence=conf_val,
                        metadata=meta_val,
                    )
                )
            except Exception as e:
                logger.debug(f"Skipping invalid memory item {p}: {e}")

        # If model extraction returned empty, supplement with heuristic extraction
        if not items:
            items = self._reflect_heuristically(conversation_events)

        return items
