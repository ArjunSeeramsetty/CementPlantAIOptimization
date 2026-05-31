# AGENTS.md - Repository Coding & Operations Guidelines

This file is a canonical repository contract for AI coding agents (such as Antigravity) working in this codebase. It documents styling rules, workspace hygiene guidelines, and operational deployment workflows.

---

## 🐍 Python Coding Standards

- **PEP 8 Compliance**: Follow PEP 8 for all generated or modified Python code.
- **Explicitness**: Prefer readable, explicit, and maintainable code over overly compact or clever syntax.
- **Documentation**: Add docstrings to all new public modules, classes, functions, and methods.
- **Type Safety**: Use type hints for all new public functions and methods where practical.
- **Patterns**: Reuse existing patterns and abstractions (under `src/cement_ai_platform`) before introducing new dependencies.
- **Single Responsibility**: Keep functions and classes focused on a single responsibility.
- **No Duplication**: Avoid duplicate logic; refactor existing helper functions rather than rewriting them.

### 📝 Logging & Diagnostics
- **Logging vs Print**: Do not use `print()` for runtime diagnostics. Always use the standard `logging` module.
- **Log Levels**:
  - `DEBUG` for verbose diagnostic states helping troubleshooting.
  - `INFO` for lifecycle milestones and process changes.
  - `WARNING` for recoverable warnings/deviations.
  - `ERROR` or `EXCEPTION` for service failures.
- **No Noisy Logs**: Avoid writing logs inside trivial helpers or tight loops.
- **Data Security**: Never log secrets, private keys, API tokens, credentials, or sensitive business payloads.

---

## 🧹 Project Structure & Hygiene

- **Modular Approach**: Follow modular programming principles. Extend existing package components (`src/cement_ai_platform`) instead of dropping ad-hoc files.
- **Smallest Change**: Make the minimal possible modification required to solve the task.
- **No Idle Documentation / Scripts**: Do not create unnecessary `.md` files, scratch scripts, demo folders, or notebooks.
- **Temporary File Lifecycle**: If scratch or validation scripts are created for debug purposes, delete them before marking tasks complete.
- **Clean Workspace**: Review files before finalizing task. Remove:
  - Temporary files and folders.
  - Unused imports and dead code blocks.
  - Duplicate setup or deploy files.

---

## ☁️ GCP Deployment & Operations

This codebase is optimized for serverless Google Cloud Platform deployment with near-zero idle cost.

### 1. GCP Project Setup (Windows PowerShell)
To configure APIs, Service Accounts, GCS Buckets, and Datasets:
```powershell
powershell -ExecutionPolicy Bypass -File deploy\setup_gcp_project.ps1
```
This script downloads credentials directly to `.secrets/cement-ops-key.json`.

### 2. Redeploying the Application
To compile, push the container via Cloud Build, and deploy to Cloud Run:
```cmd
deploy\deploy_production.bat
```

### 3. Service ON/OFF Switch (Cost Control)
We use a circuit breaker to stop and start public accessibility on Cloud Run (revokes `allUsers` run invoker binding):
```powershell
# Check current status
python manage_service.py status

# Switch OFF (Revoke public traffic, guaranteeing $0 compute cost)
python manage_service.py stop

# Switch ON (Restore public access to the live Streamlit dashboard)
python manage_service.py start
```

---

## 🤖 Antigravity Workspace Configuration
- Rules reside in: `.agent/rules/`
- Saved prompt workflows reside in: `.agent/workflows/`
- Customizations can be loaded and executed directly from the Antigravity Customization UI or triggered via `/` commands.
