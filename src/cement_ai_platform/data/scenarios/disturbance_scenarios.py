from __future__ import annotations

"""Compatibility shim for development_1 path.

Provides alias to process disturbance scenario creation utilities.
"""

from .process_disturbances import create_comprehensive_disturbance_scenarios  # noqa: F401

__all__ = ["create_comprehensive_disturbance_scenarios"]


