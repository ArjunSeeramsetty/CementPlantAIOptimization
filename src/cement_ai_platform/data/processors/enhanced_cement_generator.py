from __future__ import annotations

"""Compatibility shim for development_1 path.

Exposes `EnhancedCementDataGenerator` under
`cement_ai_platform.data.processors.enhanced_cement_generator` by importing the
canonical implementation from `data.data_pipeline.chemistry_data_generator`.
"""

from ...data.data_pipeline.chemistry_data_generator import (  # noqa: F401
    EnhancedCementDataGenerator,
)

__all__ = ["EnhancedCementDataGenerator"]


