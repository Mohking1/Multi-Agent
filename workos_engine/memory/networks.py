import uuid
from typing import Any, Optional
from config import get_config, WorkOSConfig
from workos_engine.types import MemoryItem, MemoryNetwork
from workos_engine.memory.db import MemoryDB
from workos_engine.memory.loci import SpatialLociManager
from workos_engine.memory.reflect import MemoryReflector


class CognitiveMemoryEngine:
    """
    4-Network Cognitive Memory Engine (Facts, Experiences, Entities, Beliefs)
    featuring spatial Loci domain routing and fast local FTS5 BM25 retrieval.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[WorkOSConfig] = None,
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
        metadata: Optional[dict[str, Any]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Retains an episodic experience or execution log in the Experiences network.
        """
        norm_wing, norm_hall = self.loci.normalize_locus(wing, hall)
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
        attributes: Optional[dict[str, Any]] = None,
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

        norm_wing = "people" if entity_type.lower() in ["person", "contact", "user"] else "knowledge"
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
        metadata: Optional[dict[str, Any]] = None,
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
        network: Optional[MemoryNetwork | str] = None,
        wing: Optional[str] = None,
        hall: Optional[str] = None,
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

    def get_active_beliefs(self, wing: Optional[str] = None) -> list[MemoryItem]:
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
        model_client: Optional[Any] = None,
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

    def close(self):
        """
        Closes the underlying database connection.
        """
        self.db.close()
