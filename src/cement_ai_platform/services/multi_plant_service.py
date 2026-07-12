"""Multi-plant orchestration helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


class MultiPlantService:
    """Manage tenant-level multi-plant orchestration."""

    def __init__(self, platform: Any) -> None:
        self.platform = platform

    def get_multi_plant_status(self) -> Dict[str, Any]:
        """Return the multi-plant supervisor status."""

        if self.platform.multi_plant_supervisor is None:
            return {"success": False, "error": "Multi-Plant support not available"}

        try:
            status = self.platform.multi_plant_supervisor.get_supervisor_status()
            return {"success": True, "status": status}
        except Exception as exc:
            logger.exception("Error getting multi-plant status")
            return {"success": False, "error": str(exc)}

    def start_multi_plant_orchestration(self) -> bool:
        """Start multi-plant orchestration."""

        if self.platform.multi_plant_supervisor is None:
            logger.warning("Multi-Plant support not available")
            return False

        try:
            self.platform.multi_plant_supervisor.start_orchestration()
            logger.info("Multi-plant orchestration started")
            return True
        except Exception as exc:
            logger.exception("Error starting multi-plant orchestration")
            logger.error("%s", exc)
            return False

    def stop_multi_plant_orchestration(self) -> bool:
        """Stop multi-plant orchestration."""

        if self.platform.multi_plant_supervisor is None:
            logger.warning("Multi-Plant support not available")
            return False

        try:
            self.platform.multi_plant_supervisor.stop_orchestration()
            logger.info("Multi-plant orchestration stopped")
            return True
        except Exception as exc:
            logger.exception("Error stopping multi-plant orchestration")
            logger.error("%s", exc)
            return False

    def deploy_model_to_tenant(self, tenant_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a model configuration to a tenant."""

        if self.platform.multi_plant_supervisor is None:
            return {"success": False, "error": "Multi-Plant support not available"}

        try:
            result = self.platform.multi_plant_supervisor.deploy_model_to_tenant(tenant_id, model_config)
            logger.info(
                "Model deployed to tenant %s: %s/%s",
                tenant_id,
                result.get("successful_deployments"),
                result.get("total_plants"),
            )
            return {"success": True, "deployment_result": result}
        except Exception as exc:
            logger.exception("Error deploying model to tenant %s", tenant_id)
            return {"success": False, "error": str(exc)}
