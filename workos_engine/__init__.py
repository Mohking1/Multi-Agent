"""WorkOS: Autonomous Executive AI Operating System."""

__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "ExecutivePlanner":
        from workos_engine.planner import ExecutivePlanner

        return ExecutivePlanner
    if name == "OllamaClient":
        from workos_engine.llm_client import OllamaClient

        return OllamaClient
    if name == "GeminiClient":
        from workos_engine.llm_client import GeminiClient

        return GeminiClient
    if name == "get_llm_client":
        from workos_engine.llm_client import get_llm_client

        return get_llm_client
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["ExecutivePlanner", "OllamaClient", "GeminiClient", "get_llm_client", "__version__"]
