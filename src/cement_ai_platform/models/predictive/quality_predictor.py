from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ...data.data_pipeline.chemistry_data_generator import CementQualityPredictor as _CQ


class CementQualityPredictor(_CQ):
    """Alias wrapper exposing the predictor under models.predictive."""

    pass


