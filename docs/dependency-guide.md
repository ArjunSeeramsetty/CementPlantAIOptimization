# Dependency Management & Setup Guide

This guide outlines the dependency configuration, virtual environment setup, and troubleshooting for the Cement Plant AI Digital Twin platform. All version conflicts between Streamlit, Pandas, NumPy, and GCP client libraries have been fully resolved.

---

## 🚀 Environment Setup

The platform is tested and verified on **Python 3.10+**.

### 💻 Method 1: Automated Script (Recommended)

To quickly spin up a clean virtual environment with all correct package versions, run the setup script:

**Windows (PowerShell)**:
```powershell
.\deploy\install-dependencies.ps1
```

**Linux/macOS**:
```bash
chmod +x deploy/install-dependencies.sh
./deploy/install-dependencies.sh
```

---

### 🔧 Method 2: Manual Staged Installation

If you prefer to configure your environment manually, execute the following commands in order:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 2. Upgrade core packaging tools
python -m pip install --upgrade pip setuptools wheel

# 3. Install core web & data science stack (locked versions)
pip install streamlit==1.28.2 pandas==2.0.3 numpy==1.26.4 plotly==5.17.0

# 4. Install ML & scientific computing packages
pip install scikit-learn==1.3.2 scipy==1.11.4

# 5. Install Google Cloud SDKs & AI tools
pip install google-cloud-bigquery==3.30.0 google-cloud-storage==3.9.0 google-genai==1.47.0 google-generativeai==0.8.6

# 6. Install the package in editable mode
pip install -e .
```

---

## 📊 Verified Dependency Compatibility Matrix

All components have been configured and tested to avoid dependency resolution conflicts. The final production package locks are:

| Package | Version | Compatibility Role | Status |
| :--- | :--- | :--- | :--- |
| **`streamlit`** | `1.28.2` | Core Web Dashboard | ✅ Verified |
| **`pandas`** | `2.0.3` | DataFrames & Processing | ✅ Verified |
| **`numpy`** | `1.26.4` | Matrix computations | ✅ Verified |
| **`google-cloud-bigquery`** | `3.30.0` | BigQuery ML & Data Lake | ✅ Verified |
| **`google-cloud-storage`** | `3.9.0` | Cloud Storage Storage | ✅ Verified |
| **`scipy`** | `1.11.4` | Scientific & Signal Processing | ✅ Verified |
| **`requests`** | `2.32.5` | HTTP Client | ✅ Verified |

---

## 🧪 Verification Commands

To verify that all critical imports and package versions are functional, run:

```bash
python -c "
import streamlit; print('✅ Streamlit:', streamlit.__version__)
import pandas; print('✅ Pandas:', pandas.__version__)
import numpy; print('✅ NumPy:', numpy.__version__)
import plotly; print('✅ Plotly:', plotly.__version__)
import google.cloud.bigquery; print('✅ Google Cloud BigQuery: OK')
import google.generativeai; print('✅ Google Generative AI: OK')
print('🚀 All dependencies verified successfully!')
"
```

To run the full test suite locally:
```bash
pytest -q
```

---

## 🔧 Troubleshooting

### Issue: Package version conflicts or `resolution-too-deep`
*   **Solution**: Wipe your `.venv` directory completely, create a fresh virtual environment, and use the automated setup scripts (`deploy/install-dependencies.ps1` or `deploy/install-dependencies.sh`) to force the correct installation order.

### Issue: Streamlit fails to launch
*   **Solution**: Re-install Streamlit with forced dependencies:
    ```bash
    pip install --force-reinstall streamlit==1.28.2
    ```

### Issue: Missing `pandas-gbq` warning when running pipeline
*   **Solution**: The package will automatically fall back to standard BigQuery clients. If you want direct pandas dataframe-to-table uploads, you can install the package manually:
    ```bash
    pip install pandas-gbq
    ```
