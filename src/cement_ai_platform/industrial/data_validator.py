from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SensorConstraints:
    min_value: float
    max_value: float
    max_rate_change: float  # per second
    timeout_seconds: float = 30.0


class RealTimeDataValidator:
    def __init__(self):
        self.sensor_constraints: Dict[str, SensorConstraints] = {}
        self.last_values: Dict[str, Dict[str, Any]] = {}

    def set_constraints(self, sensor_name: str, constraints: SensorConstraints) -> None:
        self.sensor_constraints[sensor_name] = constraints

    def validate(self, sensor_name: str, value: float) -> Dict[str, Any]:
        now = time.time()
        constraints = self.sensor_constraints.get(sensor_name)
        warnings: List[str] = []

        if constraints:
            if value < constraints.min_value or value > constraints.max_value:
                warnings.append(
                    f"{sensor_name} value {value} outside [{constraints.min_value}, {constraints.max_value}]"
                )

            if sensor_name in self.last_values:
                last = self.last_values[sensor_name]
                dt = max(0.001, now - float(last["timestamp"]))
                dv = abs(value - float(last["value"]))
                rate = dv / dt
                if rate > constraints.max_rate_change:
                    warnings.append(
                        f"{sensor_name} rate {rate:.2f}/s exceeds limit {constraints.max_rate_change}/s"
                    )

        self.last_values[sensor_name] = {"value": value, "timestamp": now}
        return {"valid": not warnings, "warnings": warnings, "timestamp": now}


