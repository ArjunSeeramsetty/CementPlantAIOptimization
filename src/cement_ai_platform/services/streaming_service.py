"""Streaming and Pub/Sub orchestration helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class StreamingService:
    """Manage real-time streaming state for the platform."""

    def __init__(self, platform: Any) -> None:
        self.platform = platform

    def start_realtime_stream(self, interval_seconds: int = 2) -> bool:
        """Start the simulated real-time stream."""

        if self.platform.pubsub_simulator is None:
            logger.warning("Streaming not available - install google-cloud-pubsub")
            return False

        if self.platform.streaming_active:
            logger.info("Streaming already active")
            return True

        try:
            self.platform.pubsub_simulator.start_streaming_simulation(interval_seconds)
            self.platform.streaming_active = True
            logger.info("Real-time streaming started (interval: %ss)", interval_seconds)
            return True
        except Exception as exc:
            logger.exception("Failed to start streaming")
            logger.error("%s", exc)
            return False

    def stop_realtime_stream(self) -> bool:
        """Stop the simulated real-time stream."""

        if self.platform.pubsub_simulator is None:
            return False

        if not self.platform.streaming_active:
            logger.info("Streaming not active")
            return True

        try:
            self.platform.pubsub_simulator.stop_streaming()
            self.platform.streaming_active = False
            logger.info("Real-time streaming stopped")
            return True
        except Exception as exc:
            logger.exception("Failed to stop streaming")
            logger.error("%s", exc)
            return False

    def get_realtime_stream_status(self) -> Dict[str, Any]:
        """Return the current streaming status."""

        if self.platform.pubsub_simulator is None:
            return {
                "streaming_available": False,
                "error": "google-cloud-pubsub not installed",
            }

        return {
            "streaming_available": True,
            "streaming_active": self.platform.streaming_active,
            "topics_configured": list(self.platform.pubsub_simulator.topics.keys()),
            "project_id": self.platform.pubsub_simulator.project_id,
        }

    def subscribe_to_process_data(self, callback_func: Optional[Any] = None) -> Optional[Any]:
        """Subscribe to process-variable updates."""

        if self.platform.pubsub_simulator is None:
            logger.warning("Streaming not available")
            return None

        def default_callback(data: Dict[str, Any]) -> None:
            logger.info("Process data received: Free Lime %.2f%%", data.get("free_lime_percent", 0))
            if self.platform.realtime_processor:
                self.platform.realtime_processor.process_process_variables(data)

        callback = callback_func or default_callback

        try:
            return self.platform.pubsub_simulator.subscribe_to_stream("process-variables", callback)
        except Exception as exc:
            logger.exception("Failed to subscribe to process data")
            logger.error("%s", exc)
            return None

    def subscribe_to_equipment_health(self, callback_func: Optional[Any] = None) -> Optional[Any]:
        """Subscribe to equipment-health updates."""

        if self.platform.pubsub_simulator is None:
            logger.warning("Streaming not available")
            return None

        def default_callback(data: Dict[str, Any]) -> None:
            logger.info("Equipment health data received")
            if self.platform.realtime_processor:
                self.platform.realtime_processor.process_equipment_health(data)

        callback = callback_func or default_callback

        try:
            return self.platform.pubsub_simulator.subscribe_to_stream("equipment-health", callback)
        except Exception as exc:
            logger.exception("Failed to subscribe to equipment health")
            logger.error("%s", exc)
            return None
