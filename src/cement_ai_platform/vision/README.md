# 👁️ Computer Vision Module

This directory contains the computer vision features for the **Cement Plant AI Optimization Platform**, covering perimeter safety, worker compliance auditing, and rotary kiln thermal flame segmentation.

---

## 🏗️ Current Features & Architecture

The module currently implements simulated streams and heuristic analyzers to validate local dashboard pipelines, Streamlit visualization layers, and Cloud/Vertex AI report triggers without requiring live, physical camera connections.

### 1. Perimeter Safety Monitoring (`safety_monitor.py`)
*   **Intrusion Detection**: Uses a coordinate-based intersection model to simulate a worker crossing a virtual fence boundary line (`CAM-NORTH-FENCE`).
*   **PPE Compliance Auditing**: Programmatically cycles between worker compliance states (compliant, missing hard hat, missing safety vest) to test detection triggers.
*   **AI Incident Reports**: Automatically connects to the **Gemini API (via `gemini-1.5-flash` / Vertex AI)** to compile structured, ISO 45001-compliant safety reports when a violation is flagged.
*   **Dynamic Visual Feeds**: Generates animated JPEG streams on-the-fly using standard `OpenCV` drawing algorithms to mock real-time camera telemetry.

### 2. Rotary Kiln Flame SCAN (`kiln_monitoring.py`)
*   **HSV Color-Space Segmentation**: Converts burner zone camera frames into HSV format and thresholds Hue-Saturation-Value ranges to segment different flame layers.
*   **Emitted Temperature Heuristic**: Calculates the ratio of Red, Orange, and Blue pixels to estimate the average combustion zone temperature:
    $$\text{Temp} = \frac{1450 \cdot N_{\text{blue}} + 1250 \cdot N_{\text{orange}} + 950 \cdot N_{\text{red}}}{N_{\text{total}}}$$
*   **Hotspot Analysis**: Automatically triggers shell glow warnings and flags local hotspots when estimated temperatures exceed operational safety limits ($>1475^\circ\text{C}$).

---

## 🚀 Recommended SOTA Upgrade Path

To transition the platform from simulation to live plant operations, the following SOTA (State-of-the-Art) upgrades are recommended:

### 1. High-Speed Edge Object Detection (YOLOv8 / YOLOv11)
*   **Architecture**: Deploy lightweight object detection models (specifically `YOLOv8n` or `YOLOv11n`) on edge gateway computers containing hardware accelerators (e.g., NVIDIA T4 GPUs, Jetson Orin Nano, or Edge TPUs).
*   **Benefits**:
    *   Enables real-time inference (25+ FPS) directly on edge camera streams.
    *   Drastically reduces bandwidth overhead by performing inference locally and only sending metadata/alerts to the cloud, rather than streaming raw 4K video feeds.
*   **Execution**:
    *   Install the `ultralytics` package.
    *   Fine-tune YOLO using the annotated PPE dataset loaded in `data/external/ppe_construction/`.
    *   Replace the heuristic checks in `safety_monitor.py` with standard YOLO inference loops.

### 2. Foundational Zero-Shot Kiln Segmentation (SAM)
*   **Architecture**: Transition flame contour mapping to a zero-shot foundational architecture like the **Segment Anything Model (SAM)** (or high-speed variants like `MobileSAM`).
*   **Benefits**:
    *   Unlike brittle HSV color-range thresholds or traditional GLCM texture matrices, SAM possesses global boundaries understanding.
    *   Robust to harsh industrial kiln conditions, such as camera lens obstruction from swirling clinker dust or extreme luminosity fluctuations.
*   **Execution**:
    *   Install `segment-anything` or `torchvision`.
    *   Configure box/point prompts centered around the burner zone to extract precise, dynamic flame polygons under all dust levels.

### 3. Visual Servoing via Reinforcement Learning (RL)
*   **Architecture**: Deploy an RL agent mapping flame visibility to PTZ (Pan-Tilt-Zoom) and camera exposure registers.
*   **Benefits**:
    *   Ensures that camera settings dynamically adjust to maintain a clear view of the burner zone during heavy kiln drifts, startup, or upset conditions.
*   **Execution**:
    *   Train a Deep Q-Network (DQN) in a simulated kiln camera environment.
    *   Interface the action space with standard ONVIF camera control protocols.

---

## 🛠️ Verification & Verification
To test the visual stream generation and alerting pipelines, execute:
```bash
pytest tests/test_vision_safety.py
```
To run the Streamlit dashboard rendering these views:
```bash
streamlit run src/cement_ai_platform/dashboard/unified_dashboard.py
```
