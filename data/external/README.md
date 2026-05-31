# External Datasets Directory

This directory stores external datasets used for training and evaluating our computer vision and automation models. 

To prevent bloating the git repository, this directory (`data/external/`) and the raw telemetry directory (`data/raw/`) are configured in `.gitignore` to ignore all large datasets, images, and binary files. Only this documentation file is tracked.

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

## 📥 Setup Procedure

1. Create the corresponding subdirectory under `data/external/` for the dataset you wish to load.
2. Download the dataset zip from the source URL.
3. Extract the images and labels/annotations directly into the created directory.
4. Run model training or validation scripts using the relative paths pointing to `data/external/`.
