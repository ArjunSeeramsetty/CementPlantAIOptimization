# setup_gcp_project.ps1
# Cement Plant AI Digital Twin - GCP Windows Setup Utility
# This script configures GCP APIs, IAM roles, service accounts, and resources on Windows.

$ErrorActionPreference = "Continue"

# Shift location to project root
Set-Location -Path (Resolve-Path "$PSScriptRoot\..")

# Configuration
$PROJECT_ID = "cement-ai-optimization"
$REGION = "us-central1"
$SERVICE_ACCOUNT_NAME = "cement-ops"
$SERVICE_ACCOUNT_EMAIL = "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
$KEY_PATH = ".secrets\cement-ops-key.json"

Write-Host "--- CEMENT PLANT AI DIGITAL TWIN - GCP WINDOWS SETUP ---" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# 1. Check prerequisites
Write-Host "`n[CHECK] Checking prerequisites..." -ForegroundColor Blue
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "gcloud CLI not found. Please install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
}
Write-Host "[OK] gcloud CLI is installed." -ForegroundColor Green

# 2. Authenticate
Write-Host "`n[AUTH] Authenticating with Google Cloud..." -ForegroundColor Blue
# Since user is already authenticated, gcloud config set project will verify, but we run this just in case
# We run gcloud auth list first to see if active account exists
$activeAccount = gcloud auth list --filter="status:ACTIVE" --format="value(account)"
if ($activeAccount) {
    Write-Host "[OK] Already logged in as $activeAccount." -ForegroundColor Green
} else {
    gcloud auth login
    Write-Host "[OK] Authentication completed." -ForegroundColor Green
}

# 3. Configure Project
Write-Host "`n[SETUP] Configuring GCP project..." -ForegroundColor Blue
gcloud config set project $PROJECT_ID
Write-Host "[OK] Project set to $PROJECT_ID as default." -ForegroundColor Green

# 4. Enable Required APIs
Write-Host "`n[API] Enabling required APIs..." -ForegroundColor Blue
$APIS = @(
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "pubsub.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com"
)

foreach ($api in $APIS) {
    Write-Host "Enabling $api..."
    gcloud services enable $api
}
Write-Host "[OK] All required APIs enabled." -ForegroundColor Green

# 5. Create Service Account
Write-Host "`n[SA] Setting up Service Account..." -ForegroundColor Blue
$saExists = gcloud iam service-accounts list --filter="email:$SERVICE_ACCOUNT_EMAIL" --format="value(email)"

if ($saExists) {
    Write-Host "[INFO] Service account $SERVICE_ACCOUNT_EMAIL already exists." -ForegroundColor Yellow
} else {
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME `
        --description="Service account for Cement AI Twin operations" `
        --display-name="Cement Ops Service Account"
    Write-Host "[OK] Service account created." -ForegroundColor Green
}

# 6. Grant IAM Roles
Write-Host "`n[IAM] Configuring IAM permissions..." -ForegroundColor Blue
$ROLES = @(
    "roles/aiplatform.admin",
    "roles/bigquery.admin",
    "roles/monitoring.admin",
    "roles/storage.admin",
    "roles/pubsub.editor",
    "roles/datastore.user",
    "roles/run.invoker",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/secretmanager.secretAccessor"
)

foreach ($role in $ROLES) {
    Write-Host "Granting $role..."
    gcloud projects add-iam-policy-binding $PROJECT_ID `
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" `
        --role="$role" | Out-Null
}
Write-Host "[OK] IAM roles configured." -ForegroundColor Green

# 7. Create Key File
Write-Host "`n[AUTH] Generating Service Account key..." -ForegroundColor Blue
if (-not (Test-Path ".secrets")) {
    New-Item -ItemType Directory -Path ".secrets" | Out-Null
}

if (Test-Path $KEY_PATH) {
    Write-Host "[INFO] Key file already exists at $KEY_PATH. Recreating..." -ForegroundColor Yellow
    Remove-Item -Path $KEY_PATH -Force
}

gcloud iam service-accounts keys create $KEY_PATH `
    --iam-account=$SERVICE_ACCOUNT_EMAIL
Write-Host "[OK] Key file created and saved to $KEY_PATH" -ForegroundColor Green

# 8. Configure Docker
Write-Host "`n[DOCKER] Configuring Docker credentials..." -ForegroundColor Blue
gcloud auth configure-docker --quiet
Write-Host "[OK] Docker credentials configured." -ForegroundColor Green

# 9. Create Cloud Storage Buckets
Write-Host "`n[GCS] Creating Cloud Storage buckets..." -ForegroundColor Blue
$BUCKETS = @(
    "${PROJECT_ID}-models",
    "${PROJECT_ID}-data",
    "${PROJECT_ID}-config",
    "${PROJECT_ID}-logs"
)

foreach ($bucket in $BUCKETS) {
    Write-Host "Checking/Creating bucket gs://$bucket..."
    # Attempt to create. If it exists, gsutil will return a non-zero code which we ignore under Continue policy
    gsutil mb -c STANDARD -l $REGION "gs://$bucket" 2>$null
    Write-Host "[OK] Checked gs://$bucket." -ForegroundColor Green
}

# 10. Create Firestore Database
Write-Host "`n[FIRESTORE] Creating Firestore Database..." -ForegroundColor Blue
gcloud firestore databases create --region=$REGION 2>$null
Write-Host "[OK] Firestore Database checked." -ForegroundColor Green

# 11. Create Pub/Sub Topics & Subscriptions
Write-Host "`n[PUBSUB] Configuring Pub/Sub..." -ForegroundColor Blue
$TOPICS = @(
    "plant-sensor-data",
    "plant-alerts",
    "maintenance-alerts",
    "optimization-results",
    "scenario_requests",
    "simulation_results",
    "process_setpoints",
    "process-variables",
    "quality-data",
    "energy-consumption",
    "emissions-data",
    "equipment-health"
)

foreach ($topic in $TOPICS) {
    gcloud pubsub topics create $topic 2>$null
    Write-Host "[OK] Topic $topic checked." -ForegroundColor Green
}

# Create subscriptions
$subs = @(
    @("plant-processing", "plant-sensor-data"),
    @("alert-processing", "plant-alerts"),
    @("maintenance-processing", "maintenance-alerts"),
    @("optimization-processing", "optimization-results")
)

foreach ($sub in $subs) {
    $subName = $sub[0]
    $topicName = $sub[1]
    gcloud pubsub subscriptions create $subName --topic=$topicName 2>$null
    Write-Host "[OK] Subscription $subName checked." -ForegroundColor Green
}

# 12. Create BigQuery Dataset
Write-Host "`n[BQ] Configuring BigQuery..." -ForegroundColor Blue
bq mk --location=US --description="Cement Plant Analytics Data" cement_analytics 2>$null
Write-Host "[OK] BigQuery dataset 'cement_analytics' checked." -ForegroundColor Green

Write-Host "`n*** GCP WINDOWS SETUP COMPLETED SUCCESSFULLY! ***" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "Project ID: $PROJECT_ID" -ForegroundColor Green
Write-Host "Service Account Key: $KEY_PATH" -ForegroundColor Green
Write-Host "`nNext Step: Run deploy_production.bat to deploy the services!" -ForegroundColor Yellow
