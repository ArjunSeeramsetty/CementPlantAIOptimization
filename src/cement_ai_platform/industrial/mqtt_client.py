from __future__ import annotations

import json
from typing import Any, Callable, Dict


class CementPlantMQTTClient:
    """Lightweight MQTT client wrapper with optional dependency.

    Requires `paho-mqtt` when used. Designed to avoid hard imports at module import time.
    """

    def __init__(self, broker_host: str, broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self._client = None
        self.sensor_callbacks: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("paho-mqtt not installed") from exc
        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def connect(self) -> None:
        self._ensure_client()
        assert self._client is not None
        self._client.connect(self.broker_host, self.broker_port, 60)

    def loop_start(self) -> None:
        assert self._client is not None
        self._client.loop_start()

    def loop_stop(self) -> None:
        assert self._client is not None
        self._client.loop_stop()

    def subscribe_sensor(self, sensor_topic: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to `cement_plant/sensors/<sensor_topic>` and register callback."""
        assert self._client is not None
        topic = f"cement_plant/sensors/{sensor_topic}"
        self.sensor_callbacks[topic] = callback
        self._client.subscribe(topic)

    # ------------------------------ MQTT callbacks ------------------------------
    def _on_connect(self, client, userdata, flags, rc):  # noqa: ANN001, D401 - MQTT API
        # Re-subscribe if needed on reconnect
        for topic in self.sensor_callbacks.keys():
            client.subscribe(topic)

    def _on_message(self, client, userdata, msg):  # noqa: ANN001 - MQTT API
        callback = self.sensor_callbacks.get(msg.topic)
        if not callback:
            return
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            payload = {"raw": msg.payload.decode("utf-8", errors="ignore")}
        callback(payload)


