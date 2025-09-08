from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict


class ManualOverrideSystem:
    def __init__(self) -> None:
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self.max_duration = timedelta(hours=2)

    def request(self, operator_id: str, parameter: str, value: float, duration_minutes: int = 60, reason: str = "") -> bool:
        if duration_minutes > self.max_duration.total_seconds() / 60:
            return False
        override_id = f"{parameter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.overrides[override_id] = {
            "operator_id": operator_id,
            "parameter": parameter,
            "value": value,
            "start": datetime.now(),
            "end": datetime.now() + timedelta(minutes=duration_minutes),
            "reason": reason,
            "active": True,
        }
        logging.warning("Manual override: %s => %s by %s", parameter, value, operator_id)
        return True

    def active_overrides(self) -> Dict[str, float]:
        now = datetime.now()
        result: Dict[str, float] = {}
        for data in self.overrides.values():
            if data["active"] and now < data["end"]:
                result[data["parameter"]] = float(data["value"])
            elif now >= data["end"]:
                data["active"] = False
                logging.info("Override expired: %s", data["parameter"])
        return result


