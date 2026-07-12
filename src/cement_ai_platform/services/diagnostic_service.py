"""Runtime diagnostics and validation helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# Optional imports for diagnostics components
try:
    from cement_ai_platform.maintenance.predictive_maintenance import PredictiveMaintenanceEngine
    MAINTENANCE_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    PredictiveMaintenanceEngine = None  # type: ignore[assignment]
    MAINTENANCE_AVAILABLE = False
    _maintenance_import_error = exc

try:
    from cement_ai_platform.validation.drift_detection import DataDriftDetector
    VALIDATION_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    DataDriftDetector = None  # type: ignore[assignment]
    VALIDATION_AVAILABLE = False
    _validation_import_error = exc


logger = logging.getLogger(__name__)


class DiagnosticService:
    """Coordinate diagnostics, maintenance, and drift validation."""

    def __init__(self, platform: Any) -> None:
        self.platform = platform
        self.maintenance_engine: Optional[Any] = None
        self.drift_detector: Optional[Any] = None

        if MAINTENANCE_AVAILABLE and PredictiveMaintenanceEngine is not None:
            try:
                self.maintenance_engine = PredictiveMaintenanceEngine()
            except Exception as exc:
                logger.warning("Maintenance engine initialization failed: %s", exc)

        if VALIDATION_AVAILABLE and DataDriftDetector is not None:
            try:
                self.drift_detector = DataDriftDetector()
            except Exception as exc:
                logger.warning("Data drift detector initialization failed: %s", exc)

    def detect_anomalies(self, plant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run anomaly detection using the platform's detector."""
        return self.platform.anomaly_detector.detect_anomalies(plant_data)

    def generate_maintenance_report(self, plant_id: str = "JK_Rajasthan_1", days_ahead: int = 30) -> Dict[str, Any]:
        """Generate predictive maintenance report"""
        if not self.maintenance_engine:
            return {
                'success': False,
                'error': 'Maintenance modules not available'
            }
        try:
            report = self.maintenance_engine.generate_maintenance_report(plant_id, days_ahead)
            return {
                'success': True,
                'report': report
            }
        except Exception as e:
            logger.exception("Error generating maintenance report")
            return {
                'success': False,
                'error': f'Error generating maintenance report: {str(e)}'
            }

    def predict_equipment_failure(self, equipment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict failure for specific equipment"""
        if not self.maintenance_engine:
            return {
                'success': False,
                'error': 'Maintenance modules not available'
            }
        try:
            recommendation = self.maintenance_engine.predict_equipment_failure(equipment_data)
            if recommendation:
                return {
                    'success': True,
                    'recommendation': recommendation
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to generate maintenance recommendation'
                }
        except Exception as e:
            logger.exception("Error predicting equipment failure")
            return {
                'success': False,
                'error': f'Error predicting equipment failure: {str(e)}'
            }

    def detect_data_drift(self, current_data: Any, reference_snapshot: str = "baseline") -> Dict[str, Any]:
        """Detect data drift in process variables"""
        if not self.drift_detector:
            return {
                'success': False,
                'error': 'Validation modules not available'
            }
        try:
            drift_results = self.drift_detector.detect_data_drift(current_data, reference_snapshot)
            return {
                'success': True,
                'drift_results': drift_results
            }
        except Exception as e:
            logger.exception("Error detecting data drift")
            return {
                'success': False,
                'error': f'Error detecting data drift: {str(e)}'
            }

    def create_reference_snapshot(self, data: Any, snapshot_name: str = "baseline") -> Dict[str, Any]:
        """Create reference snapshot for drift detection"""
        if not self.drift_detector:
            return {
                'success': False,
                'error': 'Validation modules not available'
            }
        try:
            success = self.drift_detector.create_reference_snapshot(data, snapshot_name)
            return {
                'success': success,
                'snapshot_name': snapshot_name
            }
        except Exception as e:
            logger.exception("Error creating reference snapshot")
            return {
                'success': False,
                'error': f'Error creating reference snapshot: {str(e)}'
            }

    def trigger_model_retraining(self, drift_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger model retraining based on drift detection"""
        if not self.drift_detector:
            return {
                'success': False,
                'error': 'Validation modules not available'
            }
        try:
            retraining_result = self.drift_detector.trigger_model_retraining(drift_summary)
            return {
                'success': True,
                'retraining_result': retraining_result
            }
        except Exception as e:
            logger.exception("Error triggering model retraining")
            return {
                'success': False,
                'error': f'Error triggering model retraining: {str(e)}'
            }
