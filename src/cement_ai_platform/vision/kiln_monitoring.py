"""Kiln monitoring scaffold using computer vision.

Integrate with Cloud Vision or on-prem edge cameras for flame analysis,
coating thickness estimation, and ring detection.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class KilnMonitoringConfig:
    frame_rate: int = 5


class KilnMonitor:
    def __init__(self, config: KilnMonitoringConfig | None = None):
        self.config = config or KilnMonitoringConfig()

    def analyze_frame(self, frame) -> dict:
        # Placeholder if no CV libs installed
        return {"flame_intensity": 0.0, "hotspots": 0}


class KilnVisionMonitor:
    """Optional Cloud Vision + OpenCV analysis. Imports lazily."""

    def __init__(self) -> None:
        self._vision = None

    def _ensure_clients(self) -> None:
        if self._vision is not None:
            return
        try:
            from google.cloud import vision  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("google-cloud-vision not installed") from exc
        self._vision = vision.ImageAnnotatorClient()

    def analyze_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        self._ensure_clients()
        from google.cloud import vision  # type: ignore
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        image = vision.Image(content=image_bytes)
        _ = self._vision.label_detection(image=image)  # Not used yet, placeholder

        np_array = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv_image, (100, 50, 50), (130, 255, 255))
        orange_mask = cv2.inRange(hsv_image, (10, 50, 50), (25, 255, 255))
        red_mask = cv2.inRange(hsv_image, (0, 50, 50), (10, 255, 255))

        blue = int(cv2.countNonZero(blue_mask))
        orange = int(cv2.countNonZero(orange_mask))
        red = int(cv2.countNonZero(red_mask))
        total = max(1, blue + orange + red)

        temperature = (blue * 1450 + orange * 1250 + red * 950) / total
        uniformity = 1.0
        try:
            import numpy as _np  # type: ignore
            uniformity = 1.0 - (float(_np.std([blue, orange, red])) / max(1.0, float(_np.mean([blue, orange, red]))))
        except Exception:
            pass

        return {
            "flame_intensity": total / float(cv_image.shape[0] * cv_image.shape[1]),
            "estimated_temperature": float(temperature),
            "flame_uniformity": max(0.0, float(uniformity)),
        }



