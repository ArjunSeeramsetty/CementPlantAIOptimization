from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List


class SafetyLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    EMERGENCY = "emergency"


class SafetyInterlockSystem:
    def __init__(self) -> None:
        self.interlocks: Dict[str, Dict[str, Any]] = {}
        self.callbacks: Dict[SafetyLevel, List[Callable[[str, Dict[str, Any]], None]]] = {
            SafetyLevel.WARNING: [],
            SafetyLevel.ALARM: [],
            SafetyLevel.EMERGENCY: [],
        }

    def on(self, level: SafetyLevel, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self.callbacks[level].append(callback)

    def register_interlock(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        level: SafetyLevel,
        action: Callable[[], None],
    ) -> None:
        self.interlocks[name] = {
            "condition": condition,
            "level": level,
            "action": action,
            "active": False,
        }

    def check(self, plant_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        triggered: List[Dict[str, Any]] = []
        for name, lock in self.interlocks.items():
            if lock["condition"](plant_state):
                if not lock["active"]:
                    lock["active"] = True
                    try:
                        lock["action"]()
                    except Exception as exc:  # pragma: no cover
                        logging.exception("Interlock action failed: %s", exc)
                    event = {"name": name, "level": lock["level"].value}
                    triggered.append(event)
                    for cb in self.callbacks.get(lock["level"], []):
                        cb(name, plant_state)
                    logging.critical("SAFETY INTERLOCK TRIGGERED: %s", name)
            else:
                lock["active"] = False
        return triggered

    def emergency_shutdown(self) -> None:
        logging.critical("EMERGENCY SHUTDOWN INITIATED")
        # Integrate with PLC to stop fuel feed, reduce kiln speed, etc.


