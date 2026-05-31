# GCP Production Deployment Guide

This guide describes how to deploy, manage, and scale the Cement Plant AI Digital Twin Platform on Google Cloud Platform with near-zero idle infrastructure cost.

---

## 🏗️ 1. GCP Project Setup

We use a fully serverless architecture (Cloud Run, BigQuery, Pub/Sub, Cloud Storage, Firestore, and Vertex AI) to avoid recurring compute costs.

To configure your GCP project automatically:
1. Open a PowerShell console.
2. Run the setup script:
   ```powershell
   powershell -ExecutionPolicy Bypass -File deploy\setup_gcp_project.ps1
   ```
This script will authenticate your Google account, enable required APIs, provision service accounts, grant IAM roles, download credentials to `.secrets/cement-ops-key.json`, and set up GCS buckets and databases.

---

## 🚀 2. Redeploying the Application

The deployment scripts automatically configure BigQuery ML models, build the container image using Google Cloud Build, push it to Artifact Registry, and deploy to Cloud Run.

### Windows (Command Prompt)
```cmd
deploy\deploy_production.bat
```

### Linux / macOS (Bash)
```bash
chmod +x deploy/deploy_production.sh
./deploy/deploy_production.sh
```

---

## 💰 3. Cost Control & Scale-to-Zero Configuration

*   **No GKE Cluster**: The GKE (Kubernetes) configurations have been disabled from the Terraform templates to avoid provisioning persistent VMs (saving ~$460/month).
*   **Scale-to-Zero**: Cloud Run is configured with `--min-instances 0` across all deploy scripts. When the system is idle, GCP automatically scales down container replicas to 0, resulting in **$0.00** compute charges.

---

## 🔌 4. Service ON/OFF Switch (Circuit Breaker)

To explicitly start or stop the public Streamlit dashboard and API access, use the `manage_service.py` utility:

```powershell
# 1. Check current status and access URL
$env:PYTHONIOENCODING="utf-8"; python manage_service.py status

# 2. Switch OFF (Blocks all public traffic, guaranteeing zero active cost)
$env:PYTHONIOENCODING="utf-8"; python manage_service.py stop

# 3. Switch ON (Grants public access)
$env:PYTHONIOENCODING="utf-8"; python manage_service.py start
```

---

## 🔐 5. GitHub Secrets Setup (CI/CD)

To configure automated builds and deployments via GitHub Actions:

1. Open your GitHub repository settings page: `https://github.com/ArjunSeeramsetty/CementPlantAIOptimization/settings`
2. Navigate to **Secrets and variables** → **Actions**.
3. Create a **New repository secret**:
   *   **Name**: `GCP_SA_KEY`
   *   **Secret**: Paste the entire contents of your local `.secrets/cement-ops-key.json` file.
4. Any push to the `main` branch will now automatically run security scans, format verification, and redeploy to Cloud Run.
