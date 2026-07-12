"""DWSIM scenario orchestration helpers."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


class SimulationService:
    """Manage DWSIM scenario execution and history retrieval."""

    def __init__(self, platform: Any) -> None:
        self.platform = platform

    def run_dwsim_scenario(self, scenario_name: str, plant_id: str = "JK_Rajasthan_1") -> Dict[str, Any]:
        """Execute a standard DWSIM scenario."""

        if self.platform.dwsim_engine is None:
            return {"success": False, "error": "DWSIM integration not available"}

        try:
            if scenario_name in self.platform.dwsim_engine.standard_scenarios:
                scenario = self.platform.dwsim_engine.standard_scenarios[scenario_name]
            else:
                return {"success": False, "error": f"Scenario {scenario_name} not found"}

            result = self.platform.dwsim_engine.execute_scenario(scenario, plant_id)
            logger.info("DWSIM scenario %s executed: %s", scenario_name, result.get("success"))
            return result
        except Exception as exc:
            logger.exception("Error executing DWSIM scenario")
            return {"success": False, "error": str(exc)}

    def create_custom_dwsim_scenario(self, scenario_config: Dict[str, Any], plant_id: str = "JK_Rajasthan_1") -> Dict[str, Any]:
        """Create and execute a custom DWSIM scenario."""

        if self.platform.dwsim_engine is None:
            return {"success": False, "error": "DWSIM integration not available"}

        try:
            from ..dwsim.dwsim_connector import DWSIMScenario

            custom_scenario = DWSIMScenario(
                scenario_id=f"custom_{int(time.time())}",
                scenario_name=scenario_config.get("name", "Custom Scenario"),
                description=scenario_config.get("description", "Custom process simulation"),
                input_parameters=scenario_config.get("parameters", {}),
                expected_outputs=scenario_config.get("outputs", ["burning_zone_temp", "free_lime_percent"]),
                simulation_duration=scenario_config.get("duration", 1800),
                priority=scenario_config.get("priority", "medium"),
            )
            result = self.platform.dwsim_engine.execute_scenario(custom_scenario, plant_id)
            logger.info("Custom DWSIM scenario executed: %s", result.get("success"))
            return result
        except Exception as exc:
            logger.exception("Error creating custom DWSIM scenario")
            return {"success": False, "error": str(exc)}

    def get_dwsim_scenario_history(self, plant_id: str = "JK_Rajasthan_1", limit: int = 20) -> List[Dict[str, Any]]:
        """Return DWSIM scenario history."""

        if self.platform.dwsim_engine is None:
            return []

        try:
            return self.platform.dwsim_engine.get_scenario_history(plant_id, limit)
        except Exception as exc:
            logger.exception("Error retrieving DWSIM scenario history")
            logger.error("%s", exc)
            return []

    def execute_dwsim_scenario(self, scenario_name: str, plant_id: str = "JK_Rajasthan_1") -> Dict[str, Any]:
        """Backward-compatible alias for run_dwsim_scenario."""

        return self.run_dwsim_scenario(scenario_name, plant_id)
