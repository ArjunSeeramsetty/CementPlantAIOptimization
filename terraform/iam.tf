# terraform/iam.tf
# Least Privilege IAM Role Bindings for Cement AI Platform

# Service Account for Cement AI Platform
resource "google_service_account" "cement_sa" {
  account_id   = "cement-ops"
  display_name = "Cement AI Platform Service Account"
  description  = "Service account for JK Cement Digital Twin Platform operations"
}

# Pub/Sub Publisher Role (for publishing sensor data)
resource "google_project_iam_member" "pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Pub/Sub Subscriber Role (for consuming messages)
resource "google_project_iam_member" "pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Logging Writer Role (for application logs)
resource "google_project_iam_member" "logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Monitoring Metric Writer Role (for custom metrics)
resource "google_project_iam_member" "monitoring_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Firestore User Role (for multi-plant data)
resource "google_project_iam_member" "firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# BigQuery Data Editor Role (for analytics data)
resource "google_project_iam_member" "bigquery_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# BigQuery Job User Role (for running queries)
resource "google_project_iam_member" "bigquery_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Storage Object Admin Role (for plant configurations)
resource "google_project_iam_member" "storage_object_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Vertex AI User Role (for AI model operations)
resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Secret Manager Secret Accessor Role (for secrets)
resource "google_project_iam_member" "secret_manager_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Cloud Trace Agent Role (for tracing)
resource "google_project_iam_member" "cloud_trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Service Account Token Creator Role (for impersonation)
resource "google_project_iam_member" "service_account_token_creator" {
  project = var.project_id
  role    = "roles/iam.serviceAccountTokenCreator"
  member  = "serviceAccount:${google_service_account.cement_sa.email}"
}

# Output the service account email for reference
output "cement_service_account_email" {
  description = "Email of the Cement AI Platform service account"
  value       = google_service_account.cement_sa.email
}

# Output the service account key (for CI/CD)
output "cement_service_account_key" {
  description = "Service account key for CI/CD"
  value       = google_service_account_key.cement_sa_key.private_key
  sensitive   = true
}

# Service Account Key for CI/CD
resource "google_service_account_key" "cement_sa_key" {
  service_account_id = google_service_account.cement_sa.name
  public_key_type    = "TYPE_X509_PEM_FILE"
}
