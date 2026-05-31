# Raw Telemetry Data Directory

This directory stores raw plant telemetry logs and high-frequency sensor streams (e.g., DCS sensor CSVs, raw kiln data).

To prevent large data payloads from bloating our Git repository history, this directory (`data/raw/`) is configured in `.gitignore` to ignore all contents. Only this documentation file is tracked.

---

## 📥 Usage Guidelines

1. Place your raw telemetry CSV files or DCS sensor log exports in this folder.
2. Training, validation, and analytics pipelines (e.g. `scripts/run_preprocess.py`) read from this folder.
3. Ensure no credentials, keys, or sensitive business payloads are saved here.
