import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from workos_engine.types import AutonomyLevel

load_dotenv()


@dataclass
class WorkOSConfig:
    # Ollama local inference & embeddings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "hf.co/unsloth/SmolLM3-3B-GGUF:UD-Q6_K_XL")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
    )

    # Autonomy & Decision Policy
    autonomy_level: AutonomyLevel = field(
        default_factory=lambda: AutonomyLevel(os.getenv("AUTONOMY_LEVEL", "SUPERVISED").upper())
    )
    trusted_recipients: list[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.getenv("AUTONOMOUS_TRUSTED_RECIPIENTS", "").split(",")
            if r.strip()
        ]
    )

    # Cognitive Memory
    memory_db_path: str = field(
        default_factory=lambda: os.getenv("WORKOS_MEMORY_DB", "workos_memory.db")
    )

    # Elasticsearch RAG
    elasticsearch_url: str = field(
        default_factory=lambda: (
            os.getenv("ELASTICSEARCH_URL")
            or os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
        )
    )
    elasticsearch_username: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_USERNAME", "elastic")
    )
    elasticsearch_password: str = field(
        default_factory=lambda: (
            os.getenv("ES_LOCAL_PASSWORD") or os.getenv("ELASTICSEARCH_PASSWORD", "")
        )
    )
    elasticsearch_api_key: str = field(
        default_factory=lambda: (
            os.getenv("ES_LOCAL_API_KEY") or os.getenv("ELASTICSEARCH_API_KEY", "")
        )
    )
    rag_index_name: str = field(
        default_factory=lambda: os.getenv("RAG_INDEX_NAME", "workos_knowledge_base")
    )

    # Email (IMAP / SMTP)
    imap_user: str = field(
        default_factory=lambda: os.getenv("IMAP_USER") or os.getenv("GMAIL_USERNAME", "")
    )
    imap_password: str = field(
        default_factory=lambda: os.getenv("IMAP_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD", "")
    )
    imap_host: str = field(
        default_factory=lambda: os.getenv(
            "IMAP_HOST",
            "imap.gmail.com" if (os.getenv("IMAP_USER") or os.getenv("GMAIL_USERNAME")) else "",
        )
    )
    imap_port: int = field(default_factory=lambda: int(os.getenv("IMAP_PORT", "993")))

    smtp_user: str = field(
        default_factory=lambda: os.getenv("SMTP_USER") or os.getenv("GMAIL_USERNAME", "")
    )
    smtp_password: str = field(
        default_factory=lambda: os.getenv("SMTP_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD", "")
    )
    smtp_host: str = field(
        default_factory=lambda: os.getenv(
            "SMTP_HOST",
            "smtp.gmail.com" if (os.getenv("SMTP_USER") or os.getenv("GMAIL_USERNAME")) else "",
        )
    )
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))


_config_instance = None


def get_config(reload: bool = False) -> WorkOSConfig:
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = WorkOSConfig()
    return _config_instance
