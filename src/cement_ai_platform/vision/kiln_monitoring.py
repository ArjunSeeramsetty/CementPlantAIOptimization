"""Kiln monitoring using computer vision.

Integrate with Cloud Vision or on-prem edge cameras for flame analysis,
coating thickness estimation, and ring detection. Also provides dynamic mock
burner zone visual simulations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import os

# Lazy load cv2 and numpy
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


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

    def generate_mock_flame_frame(self, frame_id: int, kiln_temp: float = 1450.0) -> bytes:
        """
        Generate a visual mock frame of the kiln burning zone interior.
        
        Args:
            frame_id: Counter to drive flame fluctuations.
            kiln_temp: Temperature of the kiln to influence flame colors.
        """
        if not CV2_AVAILABLE:
            return b""
            
        # Canvas: 640x360
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (10, 10, 15)  # Dark background
        
        center_x, center_y = 320, 180
        outer_radius = 160
        
        # Draw Kiln Steel Shell (Outer ring)
        cv2.circle(img, (center_x, center_y), outer_radius, (100, 100, 100), 4)
        # Refractory Brick lining (Inner ring)
        cv2.circle(img, (center_x, center_y), outer_radius - 12, (50, 70, 90), 8)
        
        # Draw heat glow on the lining (influenced by temp)
        glow_intensity = int(max(50, min(255, (kiln_temp - 1200) * 0.7)))
        cv2.circle(img, (center_x, center_y), outer_radius - 20, (0, glow_intensity // 4, glow_intensity // 2), 2)
        
        # Generate dynamic flame shape using random/sine wave perturbations
        import math
        flame_base_radius = 50 + int(math.sin(frame_id * 0.2) * 5)
        
        # Draw layers of flame (Red -> Orange -> Yellow -> Blue-white core)
        # 1. Outer Red glow
        num_points = 24
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points)
            # Add fluctuation
            r_noise = (frame_id + i) % 7
            r = flame_base_radius + 40 + r_noise + int(math.cos(angle * 3 + frame_id * 0.5) * 8)
            x = int(center_x + r * math.cos(angle))
            y = int(center_y + r * math.sin(angle))
            points.append([x, y])
        cv2.fillPoly(img, [np.array(points, dtype=np.int32)], (0, 0, 200))
        
        # 2. Middle Orange glow
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points)
            r_noise = (frame_id + i * 2) % 5
            r = flame_base_radius + 20 + r_noise + int(math.cos(angle * 4 - frame_id * 0.4) * 6)
            x = int(center_x + r * math.cos(angle))
            y = int(center_y + r * math.sin(angle))
            points.append([x, y])
        cv2.fillPoly(img, [np.array(points, dtype=np.int32)], (0, 100, 255))
        
        # 3. Inner Yellow core
        points = []
        for i in range(num_points):
            angle = i * (2 * math.pi / num_points)
            r_noise = (frame_id * 3 + i) % 4
            r = flame_base_radius + r_noise + int(math.cos(angle * 2 + frame_id * 0.6) * 4)
            x = int(center_x + r * math.cos(angle))
            y = int(center_y + r * math.sin(angle))
            points.append([x, y])
        cv2.fillPoly(img, [np.array(points, dtype=np.int32)], (0, 220, 255))
        
        # 4. White-Blue hot core (only if temp > 1400)
        if kiln_temp > 1400:
            points = []
            for i in range(num_points):
                angle = i * (2 * math.pi / num_points)
                r = flame_base_radius - 20 + int(math.sin(angle * 5 + frame_id * 0.8) * 3)
                x = int(center_x + r * math.cos(angle))
                y = int(center_y + r * math.sin(angle))
                points.append([x, y])
            cv2.fillPoly(img, [np.array(points, dtype=np.int32)], (255, 255, 220))
            
        # Draw mock scanner lines/reticle
        cv2.drawMarker(img, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        cv2.circle(img, (center_x, center_y), 30, (0, 255, 0), 1)
        
        # Text annotations
        cv2.putText(img, "KILN INTERNAL THERMAL SCAN", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"TEMP: {kiln_temp:.1f} C", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"EMISSIVITY: 0.85", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Render a simulated ring hotspot warning if temp > 1475 C
        if kiln_temp > 1475:
            cv2.putText(img, "⚠️ WARNING: HIGH SHELL GLOW", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.circle(img, (center_x + 100, center_y - 80), 15, (0, 0, 255), -1) # Red Hotspot
            cv2.putText(img, f"HOTSPOT 1", (center_x + 70, center_y - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
