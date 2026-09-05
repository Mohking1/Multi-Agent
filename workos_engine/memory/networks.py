import re
import uuid
from typing import Any

from config import WorkOSConfig, get_config
from workos_engine.memory.db import MemoryDB
from workos_engine.memory.loci import SpatialLociManager
from workos_engine.memory.reflect import MemoryReflector
from workos_engine.types import ConversationTurn, MemoryItem, MemoryNetwork


class CognitiveMemoryEngine:
    """
    4-Network Cognitive Memory Engine (Facts, Experiences, Entities, Beliefs)
    featuring spatial Loci domain routing and fast local FTS5 BM25 retrieval.
    """

    def __init__(
        self,
        db_path: str | None = None,
        config: WorkOSConfig | None = None,
    ):
        cfg = config or get_config()
        self.db_path = db_path or cfg.memory_db_path
        self.db = MemoryDB(self.db_path)
        self.loci = SpatialLociManager()
        self.reflector = MemoryReflector(self.loci)

    def retain_fact(
        self,
        wing: str,
        hall: str,
        key: str,
        content: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Retains a verified fact in the Facts network under a spatial locus.
        """
        norm_wing, norm_hall = self.loci.normalize_locus(wing, hall)
        item = MemoryItem(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            network=MemoryNetwork.FACTS,
            wing=norm_wing,
            hall=norm_hall,
            key=key.strip(),
            content=content.strip(),
            confidence=confidence,
            metadata=metadata or {},
        )
        return self.db.insert_memory(item)

    def retain_experience(
        self,
        wing: str,
        hall: str,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Retains an episodic experience or execution log in the Experiences network.
        Prevents inserting duplicate experience records with identical content.
        """
        norm_wing, norm_hall = self.loci.normalize_locus(wing, hall)
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT id FROM memories WHERE network = ? AND (key = ? OR content = ?) AND superseded_by IS NULL",
                (MemoryNetwork.EXPERIENCES.value, key.strip(), content.strip()),
            )
            existing = cursor.fetchone()
            if existing:
                return existing[0]
        except Exception:
            pass

        item = MemoryItem(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            network=MemoryNetwork.EXPERIENCES,
            wing=norm_wing,
            hall=norm_hall,
            key=key.strip(),
            content=content.strip(),
            confidence=1.0,
            metadata=metadata or {},
        )
        return self.db.insert_memory(item)

    def retain_entity(
        self,
        entity_name: str,
        entity_type: str,
        summary: str,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """
        Retains structured knowledge about a named entity.
        Upserts the entity registry and indexes it in the Entities memory network.
        """
        attrs = attributes or {}
        self.db.upsert_entity(
            entity_name=entity_name,
            entity_type=entity_type,
            summary=summary,
            attributes=attrs,
        )

        norm_wing = (
            "people" if entity_type.lower() in ["person", "contact", "user"] else "knowledge"
        )
        norm_hall = entity_type.lower().replace(" ", "_")
        key = entity_name.lower().replace(" ", "_")

        item = MemoryItem(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            network=MemoryNetwork.ENTITIES,
            wing=norm_wing,
            hall=norm_hall,
            key=key,
            content=f"{entity_name} ({entity_type}): {summary}",
            confidence=1.0,
            metadata=attrs,
        )
        return self.db.insert_memory(item)

    def retain_belief(
        self,
        wing: str,
        hall: str,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Retains a belief/preference in the Beliefs network.
        Automatically marks older active beliefs under the same key as superseded.
        """
        norm_wing, norm_hall = self.loci.normalize_locus(wing, hall)
        new_id = f"mem_{uuid.uuid4().hex[:12]}"

        # Supersede any previous active beliefs under same key (and wing/hall)
        self.db.supersede_key(
            key=key.strip(),
            wing=norm_wing,
            hall=norm_hall,
            new_id=new_id,
        )

        item = MemoryItem(
            id=new_id,
            network=MemoryNetwork.BELIEFS,
            wing=norm_wing,
            hall=norm_hall,
            key=key.strip(),
            content=content.strip(),
            confidence=1.0,
            metadata=metadata or {},
        )
        return self.db.insert_memory(item)

    def recall(
        self,
        query: str,
        network: MemoryNetwork | str | None = None,
        wing: str | None = None,
        hall: str | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """
        Recalls relevant memories using SQLite FTS5 BM25 search with spatial Loci filters.
        """
        norm_wing = wing.strip().lower() if wing else None
        norm_hall = hall.strip().lower() if hall else None
        return self.db.search_fts(
            query=query,
            network=network,
            wing=norm_wing,
            hall=norm_hall,
            include_superseded=False,
            limit=limit,
        )

    def get_active_beliefs(self, wing: str | None = None) -> list[MemoryItem]:
        """
        Retrieves all currently active, non-superseded beliefs.
        """
        norm_wing = wing.strip().lower() if wing else None
        return self.db.get_active_beliefs(wing=norm_wing)

    def get_context_index_summary(self) -> str:
        """
        Returns a low-token spatial Loci memory summary (< 200 tokens)
        for injection into agent wake-up prompt contexts.
        """
        loci_data = self.db.get_loci_structure(active_only=True)
        return self.loci.format_summary(loci_data)

    def reflect_and_update(
        self,
        conversation_events: list[Any],
        model_client: Any | None = None,
    ) -> list[str]:
        """
        Performs memory reflection over conversation history or task outputs,
        extracting and saving new items.
        """
        return self.reflector.reflect_and_update(
            db=self.db,
            conversation_events=conversation_events,
            model_client=model_client,
        )

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Records a verbatim conversation turn into the session drawer (MemPalace pattern).
        Strips any model reasoning <think> blocks before storage.
        """
        cleaned_content = content
        if isinstance(content, str):
            cleaned_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return self.db.add_conversation_turn(
            session_id=session_id, role=role, content=cleaned_content, metadata=metadata
        )

    def get_session_dialogue(self, session_id: str, limit: int = 15) -> list[ConversationTurn]:
        """
        Retrieves ordered dialogue turns for the given session.
        """
        return self.db.get_conversation_turns(session_id=session_id, limit=limit)

    def clear_session(self, session_id: str) -> int:
        """
        Clears conversation turns for a session.
        """
        return self.db.clear_session_turns(session_id=session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        Lists all active sessions with turn counts.
        """
        return self.db.list_active_sessions()

    def set_session_title(self, session_id: str, title: str) -> None:
        """
        Sets a human-readable title for a session.
        """
        self.db.set_session_title(session_id=session_id, title=title)

    def get_session_title(self, session_id: str) -> str | None:
        """
        Gets the human-readable title for a session.
        """
        return self.db.get_session_title(session_id=session_id)

    def deduplicate(self) -> int:
        """
        Cleans up duplicate memories from database.
        """
        return self.db.deduplicate_memories()

    def reflect_context(
        self,
        query: str,
        session_id: str | None = None,
        limit_memories: int = 5,
        limit_turns: int = 6,
    ) -> str:
        """
        Synthesizes narrative conversational and cognitive context (Hindsight Reflect pattern).
        Combines:
        1. Recent verbatim dialogue history from the active session.
        2. Active user beliefs & mental models.
        3. Recalled facts, entities, and past experiences.
        """
        sections: list[str] = []

        # 1. Multi-turn dialogue history
        if session_id:
            turns = self.get_session_dialogue(session_id=session_id, limit=limit_turns)
            if turns:
                dialogue_lines = [f"{t.role.capitalize()}: {t.content}" for t in turns]
                sections.append("Recent Conversation History:\n" + "\n".join(dialogue_lines))

        # 2. Active beliefs (Mental Models)
        try:
            beliefs = self.get_active_beliefs()
            if beliefs:
                belief_lines = [
                    f"- [{b.wing}/{b.hall}] {b.key}: {b.content[:150]}" for b in beliefs
                ]
                sections.append("User Beliefs & System Preferences:\n" + "\n".join(belief_lines))
        except Exception:
            pass

        # 3. Recalled facts and relevant experiences (Recall Grounding)
        try:
            recalled = self.recall(query=query, limit=limit_memories)
            if recalled:
                mem_lines = [f"- [{m.network.value}:{m.key}] {m.content[:200]}" for m in recalled]
                sections.append(
                    "Recalled Grounded Knowledge & Experiences:\n" + "\n".join(mem_lines)
                )
        except Exception:
            pass

        return "\n\n".join(sections)

    def close(self):
        """
        Closes the underlying database connection.
        """
        self.db.close()
