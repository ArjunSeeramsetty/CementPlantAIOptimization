# Cement Plant AI Optimization

Production-grade repository structure for a generative AI–driven optimization platform for cement plants. This repo organizes data pipelines, models (PINNs, optimization, generative), and cloud integrations into a coherent Python package with tests, docs, and deployment scaffolding.

## Highlights

- Organized package under `src/cement_ai_platform`
- Config management with environment variables and Google Cloud helpers
- Scaffolds for data pipelines (BigQuery, synthetic generation), processors, and models
- Integrations: Vertex AI, Gemini, Vision, Firebase
- Tests, scripts, docs, and notebooks directories

## Repository Structure

```
CementPlantAIOptimization/
├── src/cement_ai_platform/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── vertex_ai/
│   ├── gemini/
│   ├── vision/
│   └── dashboard/
├── tests/
├── scripts/
├── docs/
├── notebooks/
├── config/
├── setup.py
├── requirements.txt
└── implementation-guide.md
```

## Quickstart

1) Python setup

```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

2) Environment configuration

Create a `.env` or export environment variables (see `src/cement_ai_platform/config/settings.py`). Minimum suggested variables:

```
CEMENT_ENV=dev
CEMENT_GCP_PROJECT=your-gcp-project
CEMENT_GCP_REGION=us-central1
CEMENT_BQ_DATASET=cement_analytics
CEMENT_VERTEX_BUCKET=gs://your-staging-bucket
```

3) Run tests

```bash
pytest -q
```

4) Analyze existing UUID-named files

```bash
python scripts/analyze_uuid_files.py --root 22685ee7-c317-4938-8cd6-16009a57eb19/Development
```

See `implementation-guide.md` for a step-by-step restructuring workflow.

## Google Cloud

Authenticate with `gcloud auth application-default login` and set the project. Initialize Vertex AI via `init_vertex_ai()` from `cement_ai_platform.config.google_cloud_config`.

## License

Proprietary. All rights reserved.

