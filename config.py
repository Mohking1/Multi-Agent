import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from workos_engine.types import AutonomyLevel

load_dotenv()


@dataclass
class WorkOSConfig:
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY", "")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "gemini-3.7-flash")
    )
    autonomy_level: AutonomyLevel = field(
        default_factory=lambda: AutonomyLevel(
            os.getenv("AUTONOMY_LEVEL", "SUPERVISED").upper()
        )
    )
    trusted_recipients: list[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.getenv("AUTONOMOUS_TRUSTED_RECIPIENTS", "").split(",")
            if r.strip()
        ]
    )
    memory_db_path: str = field(
        default_factory=lambda: os.getenv("WORKOS_MEMORY_DB", "workos_memory.db")
    )
    elasticsearch_url: str = field(
        default_factory=lambda: os.getenv("ELASTICSEARCH_URL")
        or os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    )
    rag_index_name: str = field(
        default_factory=lambda: os.getenv("RAG_INDEX_NAME", "workos_knowledge_base")
    )
    imap_host: str = field(default_factory=lambda: os.getenv("IMAP_HOST", ""))
    imap_port: int = field(
        default_factory=lambda: int(os.getenv("IMAP_PORT", "993"))
    )
    imap_user: str = field(
        default_factory=lambda: os.getenv("IMAP_USER")
        or os.getenv("GMAIL_USERNAME", "")
    )
    imap_password: str = field(
        default_factory=lambda: os.getenv("IMAP_PASSWORD")
        or os.getenv("GMAIL_APP_PASSWORD", "")
    )
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = field(
        default_factory=lambda: int(os.getenv("SMTP_PORT", "587"))
    )
    smtp_user: str = field(
        default_factory=lambda: os.getenv("SMTP_USER")
        or os.getenv("GMAIL_USERNAME", "")
    )
    smtp_password: str = field(
        default_factory=lambda: os.getenv("SMTP_PASSWORD")
        or os.getenv("GMAIL_APP_PASSWORD", "")
    )


_config_instance = None


def get_config(reload: bool = False) -> WorkOSConfig:
    global _config_instance
    if _config_instance is None or reload:
        _config_instance = WorkOSConfig()
    return _config_instance
