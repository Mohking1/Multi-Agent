import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv

from workos_engine.types import AutonomyLevel

load_dotenv()


def is_test_environment() -> bool:
    """Detects if runtime is currently executing under pytest, simulation, or test mode."""
    return bool(
        os.getenv("WORKOS_TEST_MODE") == "1"
        or os.getenv("PYTEST_CURRENT_TEST")
        or "pytest" in sys.modules
    )


@dataclass
class WorkOSConfig:
    # Ollama local inference & embeddings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "refinedneuro/refinedtoolcallv5-3b")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
    )
    default_temperature: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.6"))
    )
    top_p: float = field(default_factory=lambda: float(os.getenv("TOP_P", "0.95")))
    repeat_penalty: float = field(default_factory=lambda: float(os.getenv("REPEAT_PENALTY", "1.1")))
    num_ctx: int = field(default_factory=lambda: int(os.getenv("NUM_CTX", "6144")))

    # Gemini & Cloud LLM API (optional, non-default)
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    )
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower()
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
        default_factory=lambda: (
            os.getenv("WORKOS_MEMORY_DB")
            or (
                f"/tmp/workos_test_memory_{os.getpid()}.db"
                if is_test_environment()
                else "workos_memory.db"
            )
        )
    )
    vault_dir: str = field(
        default_factory=lambda: (
            os.getenv("WORKOS_VAULT_DIR")
            or (
                f"/tmp/workos_test_vault_{os.getpid()}"
                if is_test_environment()
                else "data/vault"
            )
        )
    )
    auto_start_infrastructure: bool = field(
        default_factory=lambda: (
            os.getenv("WORKOS_AUTO_START_INFRA", "true").lower() not in ("0", "false", "no", "off")
        )
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

    # Web Intelligence & Search (Zero-Cloud SearXNG / Fallback)
    searxng_url: str = field(
        default_factory=lambda: os.getenv("SEARXNG_URL", "http://localhost:8080")
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
