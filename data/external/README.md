# External Datasets Directory

This directory stores external datasets used for training and evaluating our computer vision, process simulation, and optimization models.

To prevent bloating the Git repository, this directory (`data/external/`) and the raw telemetry directory (`data/raw/`) are configured in `.gitignore` to ignore all large datasets, images, and binary files. Only this documentation file is tracked.

---

## 🗂️ Dataset Details & Setup Instructions

### 1. PPE Kit Detection (Construction Site Workers)
*   **Target Directory**: `data/external/ppe_construction/`
*   **Source URL**: [Kaggle Dataset](https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers)
*   **Description**: High-quality curated images featuring 11 classes (Helmet, Gloves, Vest, Boots, Goggles) including explicit negative classes (no_helmet, no_gloves).
*   **Industrial Applicability**: Essential for safety cameras monitoring restricted plant access zones (e.g., preheater towers, raw mills, conveyor belts).

---

### 2. SH17 Dataset for PPE Detection (YOLO Optimized)
*   **Target Directory**: `data/external/ppe_sh17/`
*   **Source URL**: [Kaggle Code & Dataset](https://www.kaggle.com/code/mohamedhani7/ppe-detection-yolo)
*   **Description**: ~8,099 annotated images covering 17 exhaustive PPE classes including specialized safety gear (Face Shield, Respirator, Ear Protection, Harness).
*   **Industrial Applicability**: Confined Space & High-Risk Protocols. Used when maintenance crews perform refractory brick replacement inside the rotary kiln, verifying mandatory full-body harnesses and respiratory equipment.

---

### 3. Roboflow Universe: Clinker Dataset
*   **Target Directory**: `data/external/clinker_sizing/`
*   **Source URL**: [Roboflow Universe](https://universe.roboflow.com/ziix/clinker-6c23i)
*   **Description**: Open-source dataset containing labeled bounding boxes of clinker nodules.
*   **Industrial Applicability**: Clinker Granulometry & Quality. Enables cameras at the cooler outlet to detect oversized clinker "snowmen" or agglomerations that could damage the drag chain or downstream conveyor systems.

---

### 4. Cement Clinker Microstructure Dataset
*   **Target Directory**: `data/external/clinker_microstructure/`
*   **Source URL**: [ArXiv Research](https://arxiv.org/abs/2211.03223) (Search Zenodo/Kaggle for raw images)
*   **Description**: Annotated dataset of cement clinker microstructure phases.
*   **Industrial Applicability**: Lab Automation (XRF/XRD augmentation). Automates the microscopic visual evaluation of clinker alite/belite crystalline structures to predict ultimate 28-day cement compressive strength.

---

### 5. UCI Steel Industry Energy Consumption Dataset
*   **Target Directory**: `data/external/steel_energy/`
*   **Source URL**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
*   **Description**: 35,040 instances of smart-meter logs (Active Power, Lagging/Leading Reactive Power, CO2 equivalents).
*   **Industrial Applicability**: Grinding Mill Load Forecasting. Serves as a direct analogue for finish cement grinding plants to predict peak electrical loads and optimize grinding schedules around time-of-use tariffs.

---

### 6. Global CO2 Emissions from Cement Production
*   **Target Directory**: `data/external/cement_co2_emissions/`
*   **Source URL**: [Zenodo Record 20397304](https://zenodo.org/records/20397304)
*   **Description**: Annual clinker and cement process emissions data (1880-2025) distinguishing process emissions (~1.5 Gt CO2) from combustion (~0.9 Gt CO2).
*   **Industrial Applicability**: Macro-Level Energy & Emissions Forecasting. Provides baseline emissions factors for carbon tax forecasting and fuel mix modeling.

---

### 7. CEMCAP: Techno-economic evaluation of CO2 capture
*   **Target Directory**: `data/external/cemcap_co2_capture/`
*   **Source URL**: [Zenodo Record 2597091](https://zenodo.org/records/2597091)
*   **Description**: Techno-economic evaluation datasets of oxyfuel technology, chilled ammonia, and calcium looping for cement plants.
*   **Industrial Applicability**: Future State CCUS Simulation. Simulates the electrical and thermal loads required if carbon capture infrastructure is retrofitted to the kiln and preheater.

---

## 📥 Setup Procedure

1. Run the downloader script to pull all public datasets:
   ```bash
   .venv\Scripts\python.exe scripts/download_external_datasets.py
   ```
2. For the Roboflow dataset, provide your private key:
   ```bash
   .venv\Scripts\python.exe scripts/download_external_datasets.py --roboflow-key YOUR_API_KEY
   ```
3. To download the Kaggle PPE Kit manually if the anonymous downloader fails:
   * Download the zip from the Kaggle URL and extract directly into `data/external/ppe_construction/`.
