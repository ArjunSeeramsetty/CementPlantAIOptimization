#!/bin/bash

# 🏭 Cement Plant AI Digital Twin - GCP Setup Script
# This script sets up the GCP project and required services

set -e  # Exit on any error

# Configuration
PROJECT_ID="cement-ai-opt-38517"
REGION="us-central1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏭 Cement Plant AI Digital Twin - GCP Setup${NC}"
echo -e "${BLUE}===========================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Check if gcloud is installed
echo -e "\n${BLUE}📋 Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install Google Cloud SDK."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

print_status "Google Cloud SDK is installed"

# Step 2: Authenticate with Google Cloud
echo -e "\n${BLUE}🔐 Authenticating with Google Cloud...${NC}"
gcloud auth login
print_status "Authentication completed"

# Step 3: Create project (if it doesn't exist)
echo -e "\n${BLUE}🏗️  Setting up GCP project...${NC}"

# Check if project exists
if gcloud projects describe $PROJECT_ID &> /dev/null; then
    print_warning "Project $PROJECT_ID already exists"
else
    echo "Creating project $PROJECT_ID..."
    gcloud projects create $PROJECT_ID --name="Cement Plant AI Digital Twin"
    print_status "Project created successfully"
fi

# Set project as default
gcloud config set project $PROJECT_ID
print_status "Project set as default"

# Step 4: Enable billing (user needs to do this manually)
echo -e "\n${YELLOW}💳 BILLING SETUP REQUIRED${NC}"
echo "Please ensure billing is enabled for project $PROJECT_ID:"
echo "1. Go to: https://console.cloud.google.com/billing"
echo "2. Link a billing account to project: $PROJECT_ID"
echo "3. Press Enter when billing is enabled..."

read -p "Press Enter to continue after enabling billing..."

# Step 5: Enable required APIs
echo -e "\n${BLUE}🔌 Enabling required APIs...${NC}"

APIS=(
    "run.googleapis.com"
    "cloudbuild.googleapis.com"
    "containerregistry.googleapis.com"
    "bigquery.googleapis.com"
    "firestore.googleapis.com"
    "storage.googleapis.com"
    "aiplatform.googleapis.com"
    "pubsub.googleapis.com"
    "cloudfunctions.googleapis.com"
    "appengine.googleapis.com"
    "monitoring.googleapis.com"
    "logging.googleapis.com"
    "secretmanager.googleapis.com"
    "cloudresourcemanager.googleapis.com"
    "iam.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo "Enabling $api..."
    gcloud services enable $api
done

print_status "All required APIs enabled"

# Step 6: Set up App Engine (required for some services)
echo -e "\n${BLUE}🚀 Setting up App Engine...${NC}"
gcloud app create --region=$REGION 2>/dev/null || print_warning "App Engine may already exist"
print_status "App Engine configured"

# Step 7: Create service account
echo -e "\n${BLUE}👤 Creating service account...${NC}"

SERVICE_ACCOUNT="cement-app-service@$PROJECT_ID.iam.gserviceaccount.com"

# Check if service account exists
if gcloud iam service-accounts describe $SERVICE_ACCOUNT &> /dev/null; then
    print_warning "Service account already exists"
else
    gcloud iam service-accounts create cement-app-service \
        --description="Service account for Cement AI application" \
        --display-name="Cement App Service Account"
    print_status "Service account created"
fi

# Step 8: Grant necessary permissions
echo -e "\n${BLUE}🔑 Setting up IAM permissions...${NC}"

ROLES=(
    "roles/bigquery.dataEditor"
    "roles/datastore.user"
    "roles/storage.objectAdmin"
    "roles/aiplatform.user"
    "roles/pubsub.editor"
    "roles/run.invoker"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
    "roles/secretmanager.secretAccessor"
)

for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="$role" 2>/dev/null || true
done

print_status "IAM permissions configured"

# Step 9: Create and download service account key
echo -e "\n${BLUE}🔐 Creating service account key...${NC}"

if [ -f "gcp-service-account-key.json" ]; then
    print_warning "Service account key already exists"
else
    gcloud iam service-accounts keys create gcp-service-account-key.json \
        --iam-account=$SERVICE_ACCOUNT
    print_status "Service account key created: gcp-service-account-key.json"
fi

# Step 10: Configure Docker authentication
echo -e "\n${BLUE}🐳 Configuring Docker authentication...${NC}"
gcloud auth configure-docker
print_status "Docker authentication configured"

# Step 11: Set up Cloud Storage buckets
echo -e "\n${BLUE}📦 Setting up Cloud Storage buckets...${NC}"

BUCKETS=(
    "$PROJECT_ID-models"
    "$PROJECT_ID-data"
    "$PROJECT_ID-config"
    "$PROJECT_ID-logs"
)

for bucket in "${BUCKETS[@]}"; do
    gsutil mb -c STANDARD -l $REGION gs://$bucket 2>/dev/null || print_warning "Bucket $bucket may already exist"
done

print_status "Cloud Storage buckets created"

# Step 12: Set up Firestore
echo -e "\n${BLUE}🔥 Setting up Firestore database...${NC}"
gcloud firestore databases create --region=$REGION 2>/dev/null || print_warning "Firestore database may already exist"
print_status "Firestore database ready"

# Step 13: Set up Pub/Sub topics and subscriptions
echo -e "\n${BLUE}📡 Setting up Pub/Sub...${NC}"

TOPICS=(
    "plant-sensor-data"
    "plant-alerts"
    "maintenance-alerts"
    "optimization-results"
)

SUBSCRIPTIONS=(
    "plant-processing"
    "alert-processing"
    "maintenance-processing"
    "optimization-processing"
)

for topic in "${TOPICS[@]}"; do
    gcloud pubsub topics create $topic 2>/dev/null || print_warning "Topic $topic may already exist"
done

# Create subscriptions
gcloud pubsub subscriptions create plant-processing --topic=plant-sensor-data 2>/dev/null || true
gcloud pubsub subscriptions create alert-processing --topic=plant-alerts 2>/dev/null || true
gcloud pubsub subscriptions create maintenance-processing --topic=maintenance-alerts 2>/dev/null || true
gcloud pubsub subscriptions create optimization-processing --topic=optimization-results 2>/dev/null || true

print_status "Pub/Sub topics and subscriptions created"

# Step 14: Create BigQuery dataset
echo -e "\n${BLUE}📊 Setting up BigQuery dataset...${NC}"
bq mk --location=US --description="Cement Plant Analytics Data" cement_analytics 2>/dev/null || print_warning "BigQuery dataset may already exist"
print_status "BigQuery dataset created"

# Step 15: Final summary
echo -e "\n${GREEN}🎉 GCP SETUP COMPLETED SUCCESSFULLY! 🎉${NC}"
echo -e "${BLUE}===========================================${NC}"
echo -e "${GREEN}🏗️  Project ID: $PROJECT_ID${NC}"
echo -e "${GREEN}🌍 Region: $REGION${NC}"
echo -e "${GREEN}👤 Service Account: $SERVICE_ACCOUNT${NC}"
echo -e "${GREEN}🔑 Service Account Key: gcp-service-account-key.json${NC}"
echo -e "${BLUE}===========================================${NC}"

echo -e "\n${YELLOW}📋 Next Steps:${NC}"
echo -e "1. Add the service account key to GitHub Secrets as GCP_SA_KEY"
echo -e "2. Run the deployment: ./deploy.sh"
echo -e "3. Or trigger GitHub Actions deployment by pushing to main branch"

echo -e "\n${YELLOW}🔐 GitHub Secrets Setup:${NC}"
echo -e "1. Go to your GitHub repository settings"
echo -e "2. Navigate to Secrets and variables > Actions"
echo -e "3. Add new repository secret:"
echo -e "   - Name: GCP_SA_KEY"
echo -e "   - Value: Contents of gcp-service-account-key.json"

echo -e "\n${GREEN}🌟 Your GCP environment is ready for deployment!${NC}"
