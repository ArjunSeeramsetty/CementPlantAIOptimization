"""Cement AI Platform package.

This package contains modules for data pipelines, models, integrations,
and runtime configuration used to optimize cement plant operations.
"""

import logging

__all__ = [
    "config",
    "OperatorAssistant",
    "MultiObjectiveOptimizer",
    "Objective",
    "CementNSGA2Optimizer",
    "ObjectiveFn",
    "data_sourcing",
    "simulation",
    "training",
]

logger = logging.getLogger(__name__)

# Convenience re-exports for common entrypoints
try:
    from .gemini.operator_assistant import OperatorAssistant  # type: ignore
except Exception as exc:  # pragma: no cover
    logger.warning("Optional import failed for OperatorAssistant: %s", exc)

try:
    from .models.optimization import (  # type: ignore
        MultiObjectiveOptimizer,
        Objective,
        CementNSGA2Optimizer,
        ObjectiveFn,
    )
except Exception as exc:  # pragma: no cover
    logger.warning("Optional import failed for optimization exports: %s", exc)



