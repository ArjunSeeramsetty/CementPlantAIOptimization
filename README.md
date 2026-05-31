# Cement Plant AI Optimization Platform

A production-grade generative AI–driven optimization platform for cement plant operations. This platform coordinates physics-informed neural networks (PINNs), multi-objective optimization algorithms, streaming telemetry data pipelines, and serverless Google Cloud Platform (GCP) / Zerve AI integrations.

---

## 📖 Developer Documentation Index

All technical documentation has been consolidated into a structured, modular format under the [docs/](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/) directory:

- **[Dependency Resolution Guide](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/dependency-guide.md)**: Details on core environment setup, version compatibility (Streamlit, Pandas, NumPy), and package troubleshooting.
- **[Architecture & ML Features](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/architecture-and-features.md)**: Architectural specifications, predictive maintenance models, physics-informed neural networks (PINN), and streaming analytics.
- **[GCP Production Deployment](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/deployment/gcp-deployment.md)**: Instructions for compiling and deploying the serverless containerized setup on Google Cloud Run, setting up GitHub Secrets, and configuring infrastructure.
- **[Zerve AI Integration Guide](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/deployment/zerve-integration.md)**: Full details on connecting client modules with Zerve Canvas deployments, usage examples, and API definitions.

---

## 🏗️ Repository Layout

```
CementPlantAIOptimization/
├── src/cement_ai_platform/             # Core Python package
│   ├── config/                         # Environment & configuration loaders
│   ├── data/                           # Data pipelines & generators (BigQuery, synthetic)
│   ├── models/                         # ML Models & Physics-Informed Neural Networks (PINN)
│   ├── integrations/                   # Zerve AI deployment clients & connectors
│   ├── vertex_ai/                      # Vertex AI integration modules
│   ├── gemini/                         # Gemini & Generative AI modules
│   ├── vision/                         # Vision analysis models
│   └── dashboard/                      # Real-time Streamlit dashboard UI
├── tests/                              # Unit and integration test suites
├── scripts/                            # Automation and runtime scripts
├── config/                             # App configuration files (e.g. zerve_config.yml)
├── docs/                               # Modular documentation directory
│   ├── README.md                       # Sub-documentation index
│   ├── dependency-guide.md
│   ├── architecture-and-features.md
│   └── deployment/
│       ├── gcp-deployment.md
│       └── zerve-integration.md
├── deploy/                             # Cloud deployment and setup scripts
│   ├── deploy_production.bat           # Windows GCP build and deploy script
│   ├── deploy_production.sh            # Linux GCP build and deploy script
│   ├── setup_gcp_project.ps1           # GCP workspace setup script
│   └── ...
├── manage_service.py                   # Streamlit Cloud Run toggle switch (Cost control)
├── setup.py                            # Editable Python package config
├── requirements.txt                    # Main package dependencies
└── .env.example                        # Example environment variables
```

---

## ⚡ Quickstart

### 1. Python Environment Setup
We recommend setting up a virtual environment to manage dependencies:
```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Upgrade pip and install package
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration Setup
Create a `.env` file at the root of the project (referencing `.env.example` if available, or using settings in [settings.py](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/src/cement_ai_platform/config/settings.py)):
```bash
CEMENT_ENV=dev
CEMENT_GCP_PROJECT=cement-ai-optimization
CEMENT_GCP_REGION=us-central1
CEMENT_BQ_DATASET=cement_analytics
CEMENT_VERTEX_BUCKET=gs://cement-ai-optimization-staging
ZERVE_API_KEY=your_zerve_api_key_here
```

### 3. Run Verification Tests
Validate that the packages, configurations, and connections are running correctly:
```bash
# Run unit and integration tests
pytest
```

---

## 🛠️ CLI Utilities

The platform includes command-line scripts to trigger pipelines locally:
```bash
# Run preprocessing pipeline
python scripts/run_preprocess.py --input path/to/input.csv --outdir artifacts --split

# Validate preprocessed datasets
python scripts/run_validate.py --input artifacts/preprocessed.csv --outdir artifacts

# Run physical simulation generator
python scripts/run_simulate_dwsim.py --input artifacts/preprocessed.csv --outdir artifacts
```

---

## 📜 License
Proprietary. All rights reserved.
