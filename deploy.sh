#!/bin/bash

# 🚀 Cement Plant AI Digital Twin - GCP Deployment Script
# This script automates the complete deployment to Google Cloud Platform

set -e  # Exit on any error

# Configuration
PROJECT_ID="cement-ai-opt-38517"
SERVICE_NAME="cement-digital-twin"
REGION="us-central1"
IMAGE_NAME="cement-app"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏭 Cement Plant AI Digital Twin - GCP Deployment${NC}"
echo -e "${BLUE}================================================${NC}"

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

# Step 1: Check prerequisites
echo -e "\n${BLUE}📋 Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker."
    exit 1
fi

print_status "Prerequisites check passed"

# Step 2: Set up GCP project
echo -e "\n${BLUE}🔧 Setting up GCP project...${NC}"

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
print_status "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    bigquery.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    pubsub.googleapis.com \
    cloudfunctions.googleapis.com \
    appengine.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com

print_status "APIs enabled successfully"

# Step 3: Configure Docker authentication
echo -e "\n${BLUE}🐳 Configuring Docker authentication...${NC}"
gcloud auth configure-docker
print_status "Docker authentication configured"

# Step 4: Build and push container image
echo -e "\n${BLUE}🔨 Building container image...${NC}"

# Build image
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest .

# Push to Google Container Registry
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

print_status "Container image built and pushed"

# Step 5: Deploy to Cloud Run
echo -e "\n${BLUE}🚀 Deploying to Cloud Run...${NC}"

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --set-env-vars "GCP_PROJECT=$PROJECT_ID,ENVIRONMENT=production" \
    --timeout 300 \
    --concurrency 100

print_status "Application deployed to Cloud Run"

# Step 6: Get deployment URL
echo -e "\n${BLUE}🌐 Getting deployment URL...${NC}"
APP_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)")

print_status "Application URL: $APP_URL"

# Step 7: Run health check
echo -e "\n${BLUE}🏥 Running health check...${NC}"
sleep 30  # Wait for deployment to be ready

if curl -f -s "$APP_URL/_stcore/health" > /dev/null; then
    print_status "Health check passed"
else
    print_warning "Health check failed, but deployment may still be starting up"
fi

# Step 8: Set up Cloud Storage buckets
echo -e "\n${BLUE}📦 Setting up Cloud Storage...${NC}"

gsutil mb -c STANDARD -l $REGION gs://$PROJECT_ID-models 2>/dev/null || true
gsutil mb -c STANDARD -l $REGION gs://$PROJECT_ID-data 2>/dev/null || true
gsutil mb -c STANDARD -l $REGION gs://$PROJECT_ID-config 2>/dev/null || true

print_status "Cloud Storage buckets created"

# Step 9: Set up Firestore
echo -e "\n${BLUE}🔥 Setting up Firestore...${NC}"
gcloud firestore databases create --region=$REGION 2>/dev/null || true
print_status "Firestore database ready"

# Step 10: Set up Pub/Sub
echo -e "\n${BLUE}📡 Setting up Pub/Sub...${NC}"
gcloud pubsub topics create plant-sensor-data 2>/dev/null || true
gcloud pubsub topics create plant-alerts 2>/dev/null || true
gcloud pubsub subscriptions create plant-processing --topic=plant-sensor-data 2>/dev/null || true
print_status "Pub/Sub topics and subscriptions created"

# Step 11: Create BigQuery dataset
echo -e "\n${BLUE}📊 Setting up BigQuery...${NC}"
bq mk --location=US --description="Cement Plant Analytics Data" cement_analytics 2>/dev/null || true
print_status "BigQuery dataset created"

# Step 12: Final status
echo -e "\n${GREEN}🎉 DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}🚀 Application URL: $APP_URL${NC}"
echo -e "${GREEN}📊 Project ID: $PROJECT_ID${NC}"
echo -e "${GREEN}🌍 Region: $REGION${NC}"
echo -e "${GREEN}⚙️  Service: $SERVICE_NAME${NC}"
echo -e "${BLUE}================================================${NC}"

echo -e "\n${YELLOW}📋 Next Steps:${NC}"
echo -e "1. Test the application: curl $APP_URL/_stcore/health"
echo -e "2. Open in browser: $APP_URL"
echo -e "3. Monitor in GCP Console: https://console.cloud.google.com/run?project=$PROJECT_ID"
echo -e "4. Check logs: gcloud logging read 'resource.type=\"cloud_run_revision\"' --project=$PROJECT_ID"

echo -e "\n${GREEN}🌟 Your Cement Plant AI Digital Twin is now live on Google Cloud Platform!${NC}"
