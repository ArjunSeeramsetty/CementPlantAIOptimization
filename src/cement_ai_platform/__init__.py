"""Cement AI Platform package.

This package contains modules for data pipelines, models, integrations,
and runtime configuration used to optimize cement plant operations.
"""

__all__ = [
    "config",
    "OperatorAssistant",
    "MultiObjectiveOptimizer",
    "Objective",
    "CementNSGA2Optimizer",
    "ObjectiveFn",
]

# Convenience re-exports for common entrypoints
try:
    from .gemini.operator_assistant import OperatorAssistant  # type: ignore
except Exception:  # pragma: no cover
    pass

try:
    from .models.optimization import (  # type: ignore
        MultiObjectiveOptimizer,
        Objective,
        CementNSGA2Optimizer,
        ObjectiveFn,
    )
except Exception:  # pragma: no cover
    pass



