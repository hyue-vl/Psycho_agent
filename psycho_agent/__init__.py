"""Public package surface for the Psycho-World multi-agent system."""

try:
    from .workflow import PsychoWorldGraph
except ModuleNotFoundError:  # pragma: no cover - optional dependency during tests
    PsychoWorldGraph = None  # type: ignore[assignment]

__all__ = ["PsychoWorldGraph"]
