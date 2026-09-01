"""WorkOS: Autonomous Executive AI Operating System."""

__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "ExecutivePlanner":
        from workos_engine.planner import ExecutivePlanner
        return ExecutivePlanner
    if name == "OllamaClient":
        from workos_engine.llm_client import OllamaClient
        return OllamaClient
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["ExecutivePlanner", "OllamaClient", "__version__"]
