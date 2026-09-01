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
                key = "summary_format" if "format" in pref_text.lower() else "user_preference"
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

        return items

    def _reflect_with_model(
        self, conversation_events: list[Any], model_client: Any
    ) -> list[MemoryItem]:
        """
        Uses Ollama LLM to extract structured memories.
        """
        events_json = json.dumps(
            [e if isinstance(e, (dict, str)) else str(e) for e in conversation_events]
        )
        prompt = f"""
Analyze the following conversation events and extract structured memory items to retain.
Categorize each into one of 4 networks:
- 'facts': Permanent factual knowledge (people's roles, system configs, facts)
- 'experiences': Specific episodic execution logs or task results
- 'entities': Named entities (people, tools, projects)
- 'beliefs': Working assumptions, user preferences, formatting rules

Spatial Loci organization:
- wing: 'people', 'projects', 'workflows', 'knowledge', or 'general'
- hall: specific room (e.g., 'contacts', 'decisions', 'preferences', 'runs')

Events:
{events_json}

Return ONLY valid JSON array with objects matching:
[
  {{
    "network": "facts|experiences|entities|beliefs",
    "wing": "...",
    "hall": "...",
    "key": "...",
    "content": "...",
    "confidence": 1.0,
    "metadata": {{}}
  }}
]
"""
        if hasattr(model_client, "generate"):
            resp_text = model_client.generate(prompt=prompt, format="json").strip()
        elif hasattr(model_client, "models"):
            response = model_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            resp_text = response.text.strip()
        else:
            return self._reflect_rule_based(conversation_events)

        # Clean potential markdown code fences
        if resp_text.startswith("```"):
            resp_text = re.sub(r"^```(?:json)?\n?", "", resp_text)
            resp_text = re.sub(r"\n?```$", "", resp_text)

        parsed = json.loads(resp_text)
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
                network_val = p["network"].lower().strip()
                items.append(
                    MemoryItem(
                        id=f"mem_{uuid.uuid4().hex[:12]}",
                        network=MemoryNetwork(network_val),
                        wing=p.get("wing", "general"),
                        hall=p.get("hall", "general"),
                        key=p.get("key", "item"),
                        content=p.get("content", ""),
                        confidence=float(p.get("confidence", 1.0)),
                        metadata=p.get("metadata", {}),
                    )
                )
            except Exception as e:
                logger.debug(f"Skipping invalid memory item {p}: {e}")
        return items
