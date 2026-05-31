"""Computer Vision Safety Monitoring System.

Provides boundary wall jump/intrusion detection, PPE compliance checking, 
and integration with Gemini for generating safety reports.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Lazy load dependencies to ensure dashboard doesn't crash if they are installing
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import google.generativeai as genai
    from vertexai.generative_models import GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class SafetyMonitor:
    """Handles perimeter safety and PPE compliance analysis."""

    def __init__(self, project_id: str = None):
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT', 'cement-ai-optimization')
        self._initialize_gemini()

    def _initialize_gemini(self) -> None:
        """Initialize Gemini clients for report generation."""
        self.gemini_model = None
        
        # Check standard google-generativeai API key first
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                print("✅ SafetyMonitor: Gemini initialized via API Key")
                return
            except Exception as e:
                print(f"⚠️ SafetyMonitor: API Key initialization failed: {e}")

        # Fallback to Vertex AI if available
        if GEMINI_AVAILABLE:
            try:
                import vertexai
                vertexai.init(project=self.project_id, location="us-central1")
                self.gemini_model = GenerativeModel("gemini-1.5-flash")
                print("✅ SafetyMonitor: Gemini initialized via Vertex AI")
            except Exception as e:
                print(f"⚠️ SafetyMonitor: Vertex AI initialization failed: {e}")

    def check_boundary_violation(self, frame_id: int) -> Dict[str, Any]:
        """
        Simulate perimeter security camera feed and check for boundary crossings.
        
        Args:
            frame_id: Incremental frame counter to drive mock movements.
        """
        # Define boundary line coordinates (x1, y1) to (x2, y2)
        boundary_line = ((100, 300), (500, 300))
        
        # Simulate worker walking towards and crossing the fence
        # Y position moves from 220 to 380 over 60 frames
        cycle_position = (frame_id % 120)
        if cycle_position < 60:
            y_pos = 220 + int(cycle_position * (160 / 60.0))
        else:
            y_pos = 380 - int((cycle_position - 60) * (160 / 60.0))
            
        worker_pos = (300, y_pos)
        
        # Check if crossed the y=300 boundary
        is_crossing = y_pos > 300
        
        detections = []
        if is_crossing:
            detections.append({
                "label": "Intruder (Fence Crossed)",
                "bbox": (worker_pos[0] - 25, worker_pos[1] - 50, worker_pos[0] + 25, worker_pos[1] + 50),
                "severity": "CRITICAL",
                "timestamp": datetime.now().isoformat()
            })
        else:
            detections.append({
                "label": "Person (Secure Zone)",
                "bbox": (worker_pos[0] - 25, worker_pos[1] - 50, worker_pos[0] + 25, worker_pos[1] + 50),
                "severity": "NORMAL",
                "timestamp": datetime.now().isoformat()
            })
            
        return {
            "boundary_line": boundary_line,
            "is_violation": is_crossing,
            "detections": detections,
            "camera_id": "CAM-NORTH-FENCE"
        }

    def check_ppe_compliance(self, frame_id: int) -> Dict[str, Any]:
        """
        Simulate loading bay camera and check for hard hat / high-vis compliance.
        """
        # Toggle compliance status every 100 frames for demonstration
        cycle = (frame_id // 50) % 3
        
        detections = []
        if cycle == 0:
            # Fully compliant worker
            detections.append({
                "label": "Worker-01",
                "bbox": (150, 100, 250, 350),
                "ppe": {
                    "hard_hat": True,
                    "safety_vest": True
                },
                "compliant": True
            })
        elif cycle == 1:
            # Missing hard hat
            detections.append({
                "label": "Worker-02",
                "bbox": (150, 100, 250, 350),
                "ppe": {
                    "hard_hat": False,
                    "safety_vest": True
                },
                "compliant": False,
                "violation": "MISSING HARD HAT"
            })
        else:
            # Missing safety vest
            detections.append({
                "label": "Worker-03",
                "bbox": (150, 100, 250, 350),
                "ppe": {
                    "hard_hat": True,
                    "safety_vest": False
                },
                "compliant": False,
                "violation": "MISSING SAFETY VEST"
            })

        return {
            "detections": detections,
            "is_violation": any(not d["compliant"] for d in detections),
            "camera_id": "CAM-LOADING-BAY-3"
        }

    def generate_safety_incident_report(self, camera_id: str, incident_type: str, details: str) -> str:
        """
        Generate a structured incident report using Gemini.
        Falls back to local template if Gemini is unavailable.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""
        Generate an official, professional Industrial Safety Incident Report for JK Cement.
        
        Incident Details:
        - Timestamp: {timestamp}
        - Camera ID: {camera_id}
        - Incident Type: {incident_type}
        - Observations: {details}
        
        Format the report with the following sections:
        1. INCIDENT OVERVIEW (Summarize what happened)
        2. COMPLIANCE ASSESSMENT (Reference standard ISO 45001 / OHSAS 18001 guidelines)
        3. CORRECTIVE ACTIONS (Provide 3 immediate operational recommendations)
        4. SAFETY PROTOCOL REVIEW (Propose adjustments for long-term prevention)
        """
        
        if self.gemini_model:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                return self._fallback_incident_report(timestamp, camera_id, incident_type, f"{details} (Gemini Error: {e})")
        else:
            return self._fallback_incident_report(timestamp, camera_id, incident_type, details)

    def _fallback_incident_report(self, timestamp: str, camera_id: str, incident_type: str, details: str) -> str:
        """Fallback local generation in case Gemini is offline."""
        return f"""### ⚠️ INDUSTRIAL SAFETY INCIDENT REPORT
**JK Cement Plant Operations**

**1. INCIDENT OVERVIEW**
*   **Incident Type**: {incident_type}
*   **Location/Camera**: {camera_id}
*   **Date/Time**: {timestamp}
*   **Status**: ACTIVE ALERT - UNDER INVESTIGATION
*   **Observations**: {details}

**2. COMPLIANCE ASSESSMENT**
*   **Standard Violated**: ISO 45001 (Occupational Health & Safety) Section 6.1 / Plant Safety Code Section 4.2.
*   **Evaluation**: Operational zone security breach detected. Visual validation confirms physical presence in restricted boundary or missing PPE safety barriers.

**3. CORRECTIVE ACTIONS**
1.  Dispatch local security/shift supervisor immediately to resolve the breach zone.
2.  Log incident ID in the plant safety registry and trigger local warning speakers.
3.  Perform immediate safety tool-box talk for loading bay crews.

**4. SAFETY PROTOCOL REVIEW**
*   Inspect boundary physical fence integrity within 24 hours.
*   Retrain operators on mandatory PPE enforcement zones.
"""

    def generate_mock_frame_bytes(self, frame_type: str, frame_id: int) -> bytes:
        """
        Generate a visual mock security frame as image bytes using OpenCV.
        
        Args:
            frame_type: 'boundary' or 'ppe'
            frame_id: Frame counter for animation
        """
        if not CV2_AVAILABLE:
            # Fallback black image if OpenCV not available
            return b""
            
        # Create a blank dark gray canvas (640x360)
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)  # Dark charcoal background
        
        # Draw camera overlay details
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
        cv2.putText(img, f"REC [●] {timestamp}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if frame_type == 'boundary':
            cv2.putText(img, "CAM-NORTH-FENCE - BOUNDARY SAFETY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Draw grid lines for industrial interface look
            for x in range(0, 640, 80):
                cv2.line(img, (x, 0), (x, 360), (45, 45, 45), 1)
            for y in range(0, 360, 60):
                cv2.line(img, (0, y), (640, y), (45, 45, 45), 1)
                
            # Check violation status
            status = self.check_boundary_violation(frame_id)
            is_violation = status["is_violation"]
            
            # Draw boundary line (Yellow if secure, Red if violated)
            line_color = (0, 0, 255) if is_violation else (0, 255, 255)
            x1, y1 = status["boundary_line"][0]
            x2, y2 = status["boundary_line"][1]
            cv2.line(img, (x1, y1), (x2, y2), line_color, 3)
            cv2.putText(img, "WARNING: PERIMETER LIMIT", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)
            
            # Draw simulated worker
            detection = status["detections"][0]
            bx1, by1, bx2, by2 = detection["bbox"]
            box_color = (0, 0, 255) if is_violation else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(img, (bx1, by1), (bx2, by2), box_color, 2)
            cv2.putText(img, detection["label"], (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            
            # Draw a simplified head/body outline for the person
            head_center = ((bx1 + bx2) // 2, by1 + 15)
            cv2.circle(img, head_center, 10, (150, 150, 150), -1)
            cv2.line(img, (head_center[0], head_center[1] + 10), (head_center[0], by2 - 10), (150, 150, 150), 3)
            
            # Highlight breach alert
            if is_violation:
                cv2.rectangle(img, (0, 0), (640, 360), (0, 0, 255), 4) # Red screen flash
                cv2.putText(img, "ALARM - PERIMETER BREACH", (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        elif frame_type == 'ppe':
            cv2.putText(img, "CAM-LOADING-BAY-3 - PPE COMPLIANCE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Fetch PPE status
            status = self.check_ppe_compliance(frame_id)
            detection = status["detections"][0]
            bx1, by1, bx2, by2 = detection["bbox"]
            has_hat = detection["ppe"]["hard_hat"]
            has_vest = detection["ppe"]["safety_vest"]
            is_compliant = detection["compliant"]
            
            box_color = (0, 255, 0) if is_compliant else (0, 0, 255)
            
            # Draw worker outline
            head_center = ((bx1 + bx2) // 2, by1 + 50)
            cv2.circle(img, head_center, 25, (180, 180, 180), -1) # Head
            cv2.rectangle(img, (bx1 + 10, head_center[1] + 25), (bx2 - 10, by2 - 10), (120, 120, 120), -1) # Torso
            
            # Draw Hard Hat if present, else highlight bare head
            if has_hat:
                # Yellow hard hat
                cv2.ellipse(img, (head_center[0], head_center[1] - 15), (28, 12), 0, 180, 360, (0, 255, 255), -1)
                cv2.line(img, (head_center[0] - 32, head_center[1] - 15), (head_center[0] + 32, head_center[1] - 15), (0, 255, 255), 3)
            else:
                # Missing label overlay on head
                cv2.putText(img, "NO HAT!", (head_center[0] - 25, head_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            # Draw Safety Vest if present
            if has_vest:
                # Orange safety vest overlay
                cv2.rectangle(img, (bx1 + 15, head_center[1] + 30), (bx2 - 15, by2 - 30), (0, 165, 255), -1)
                # Reflective stripes
                cv2.line(img, (bx1 + 25, head_center[1] + 30), (bx1 + 25, by2 - 30), (200, 255, 200), 2)
                cv2.line(img, (bx2 - 25, head_center[1] + 30), (bx2 - 25, by2 - 30), (200, 255, 200), 2)
            else:
                # Missing label overlay on torso
                cv2.putText(img, "NO VEST!", (head_center[0] - 30, head_center[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
            # Draw bounding box
            cv2.rectangle(img, (bx1, by1), (bx2, by2), box_color, 2)
            
            # Label
            label_text = f"{detection['label']} (COMPLIANT)" if is_compliant else f"{detection['label']} - {detection['violation']}"
            cv2.putText(img, label_text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            
            # Draw compliance status card in top right
            card_color = (0, 200, 0) if is_compliant else (0, 0, 200)
            cv2.rectangle(img, (480, 20), (620, 80), card_color, -1)
            cv2.putText(img, "PPE SYSTEM", (490, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            status_text = "PASS" if is_compliant else "BREACH"
            cv2.putText(img, status_text, (490, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        # Encode image as JPEG bytes
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()
