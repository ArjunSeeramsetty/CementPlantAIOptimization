# Cement Plant AI Optimization - Documentation Index

Welcome to the technical documentation directory. This folder contains structured, modular guides for configuring, developing, and operating the platform.

---

## 📂 Documentation Layout

- **[Dependency Resolution Guide](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/dependency-guide.md)**
  - Python environment requirements
  - Version matrices (NumPy, Pandas, Streamlit)
  - Virtual environments (`venv`, `uv`)
  - Resolving package conflicts & troubleshooting
- **[Architecture & ML Features](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/architecture-and-features.md)**
  - Digital twin concepts and physical/chemical constraints
  - Physics-Informed Neural Networks (PINN) for free lime estimation
  - Predictive maintenance classifier architectures
  - Real-time optimization solvers
  - Real-time telemetry and streaming data pipelines
- **[GCP Production Deployment](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/deployment/gcp-deployment.md)**
  - Architecture overview of Google Cloud Run and Firebase integrations
  - Compilation & packaging with GCP Cloud Build
  - Deployment execution scripts (`deploy_production.bat`, `deploy_production.sh`)
  - Automated deployment workflows via GitHub Actions
  - Environment secrets configuration reference
- **[Zerve AI Integration Guide](file:///c:/Users/arjun/Desktop/CementPlantAIOptimization/docs/deployment/zerve-integration.md)**
  - Zerve Canvas deployment-based architecture overview
  - Client modules and convenience functions reference
  - Sample workflows (Energy optimization, maintenance prediction)
  - Configuration settings and environment variables
  - Connection troubleshooting and health verification

---

## 🛠️ Operations & Development Quickstart

To quickly verify your configuration:
1. Ensure your environment variables are configured in `.env`.
2. Run standard diagnostic tests:
   ```bash
   pytest
   ```
3. Test GCP configuration status:
   ```powershell
   python manage_service.py status
   ```
