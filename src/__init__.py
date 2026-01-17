"""Clinical Triage Router - SLM Agent for Patient Intake Classification."""

__version__ = "0.1.0"

# Lazy imports - modules are available but won't fail if deps missing
__all__ = ["config", "schemas", "data_generator", "trainer", "agent", "evaluator"]


def __getattr__(name):
    """Lazy import modules on first access."""
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

