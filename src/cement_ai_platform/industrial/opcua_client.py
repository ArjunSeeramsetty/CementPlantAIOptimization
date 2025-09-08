from __future__ import annotations

from typing import Any, Dict, List


class CementPlantOPCUAClient:
    """Minimal OPC-UA client wrapper with lazy import.

    Requires `opcua` (python-opcua) to be installed when used.
    """

    def __init__(self, opc_url: str):
        self.opc_url = opc_url
        self._client = None
        self.connected = False

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from opcua import Client  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("python-opcua is not installed") from exc
        self._client = Client(self.opc_url)

    def connect(self) -> None:
        self._ensure_client()
        assert self._client is not None
        self._client.connect()
        self.connected = True

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self.connected = False

    def read_sensor_values(self, node_ids: List[str]) -> Dict[str, Any]:
        if not self.connected or self._client is None:
            raise RuntimeError("OPC-UA client not connected")
        values: Dict[str, Any] = {}
        for node_id in node_ids:
            try:
                node = self._client.get_node(node_id)
                values[node_id] = node.get_value()
            except Exception as exc:
                values[node_id] = {"error": str(exc)}
        return values

    def write_setpoint(self, node_id: str, value: float) -> bool:
        if not self.connected or self._client is None:
            raise RuntimeError("OPC-UA client not connected")
        try:
            node = self._client.get_node(node_id)
            node.set_value(value)
            return True
        except Exception:
            return False


