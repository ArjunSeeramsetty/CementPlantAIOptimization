from __future__ import annotations

"""Compatibility shim for development_1 path.

Exposes `CementChemistry` under
`cement_ai_platform.data.processors.cement_chemistry_core` by importing the
canonical implementation from `data.data_pipeline.chemistry_data_generator`.
"""

from ...data.data_pipeline.chemistry_data_generator import CementChemistry  # noqa: F401

__all__ = ["CementChemistry"]


