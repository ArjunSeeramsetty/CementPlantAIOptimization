from __future__ import annotations

"""Compatibility shim for development_1 path.

Re-exports `CementEnergyPredictor` for frameworks expecting
`models.predictive.energy_framework`.
"""

from .energy_predictor import CementEnergyPredictor  # noqa: F401

__all__ = ["CementEnergyPredictor"]


