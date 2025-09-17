terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "cement-plant-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "cement-ai-opt-38517"
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "container.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com"
  ])
  
  service = each.value
  project = var.project_id
  
  disable_dependent_services = false
}

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Cloud Storage Buckets
resource "google_storage_bucket" "model_artifacts" {
  name     = "cement-plant-model-artifacts-${random_string.suffix.result}"
  location = "US"
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
  
  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "terraform_state" {
  name     = "cement-plant-terraform-state-${random_string.suffix.result}"
  location = "US"
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  depends_on = [google_project_service.apis]
}

# BigQuery Dataset
resource "google_bigquery_dataset" "cement_analytics" {
  dataset_id                  = "cement_analytics"
  friendly_name              = "Cement Plant Analytics"
  description                = "Dataset for cement plant digital twin analytics"
  location                   = "US"
  default_table_expiration_ms = 3600000
  
  access {
    role          = "OWNER"
    user_by_email = "cement-ops@cement-ai-opt-38517.iam.gserviceaccount.com"
  }
  
  depends_on = [google_project_service.apis]
}

# Create sample tables for BigQuery ML
resource "google_bigquery_table" "process_variables" {
  dataset_id = google_bigquery_dataset.cement_analytics.dataset_id
  table_id   = "process_variables"
  
  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "feed_rate_tph"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "fuel_rate_tph"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "burning_zone_temp_c"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "kiln_speed_rpm"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "raw_meal_fineness"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "free_lime_percent"
      type = "FLOAT"
      mode = "NULLABLE"
    }
  ])
  
  depends_on = [google_bigquery_dataset.cement_analytics]
}

resource "google_bigquery_table" "energy_consumption" {
  dataset_id = google_bigquery_dataset.cement_analytics.dataset_id
  table_id   = "energy_consumption"
  
  schema = jsonencode([
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "feed_rate_tph"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "fuel_rate_tph"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "kiln_speed_rpm"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "preheater_stage1_temp_c"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "preheater_stage2_temp_c"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "preheater_stage3_temp_c"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "o2_percent"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "thermal_energy_kcal_kg"
      type = "FLOAT"
      mode = "NULLABLE"
    }
  ])
  
  depends_on = [google_bigquery_dataset.cement_analytics]
}

# GKE Cluster for production deployment
resource "google_container_cluster" "cement_plant_cluster" {
  name     = "cement-plant-cluster"
  location = var.region
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  depends_on = [google_project_service.apis]
}

resource "google_container_node_pool" "cement_plant_nodes" {
  name       = "cement-plant-node-pool"
  location   = var.region
  cluster    = google_container_cluster.cement_plant_cluster.name
  node_count = 3
  
  node_config {
    preemptible  = false
    machine_type = "e2-standard-4"
    
    service_account = "cement-ops@cement-ai-opt-38517.iam.gserviceaccount.com"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
  
  autoscaling {
    min_node_count = 3
    max_node_count = 10
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Artifact Registry for container images
resource "google_artifact_registry_repository" "cement_plant_repo" {
  location      = var.region
  repository_id = "cement-plant-repo"
  description   = "Docker repository for cement plant digital twin"
  format        = "DOCKER"
  
  depends_on = [google_project_service.apis]
}

# Cloud Run service configuration
resource "google_cloud_run_v2_service" "cement_plant_service" {
  name     = "cement-plant-digital-twin"
  location = var.region
  
  template {
    scaling {
      min_instance_count = 2
      max_instance_count = 100
    }
    
    containers {
      image = "gcr.io/${var.project_id}/cement-plant-ai:latest"
      
      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }
      
      env {
        name  = "CEMENT_ENV"
        value = "production"
      }
      
      env {
        name  = "CEMENT_GCP_PROJECT"
        value = var.project_id
      }
      
      env {
        name  = "CEMENT_BQ_DATASET"
        value = "cement_analytics"
      }
    }
    
    service_account = "cement-ops@cement-ai-opt-38517.iam.gserviceaccount.com"
  }
  
  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access to Cloud Run service
resource "google_cloud_run_service_iam_member" "cement_plant_public_access" {
  location = google_cloud_run_v2_service.cement_plant_service.location
  service  = google_cloud_run_v2_service.cement_plant_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Outputs
output "cluster_endpoint" {
  value = google_container_cluster.cement_plant_cluster.endpoint
}

output "cluster_ca_certificate" {
  value = google_container_cluster.cement_plant_cluster.master_auth[0].cluster_ca_certificate
}

output "model_artifacts_bucket" {
  value = google_storage_bucket.model_artifacts.name
}

output "bigquery_dataset" {
  value = google_bigquery_dataset.cement_analytics.dataset_id
}

output "cloud_run_url" {
  value = google_cloud_run_v2_service.cement_plant_service.uri
}

output "artifact_registry_url" {
  value = google_artifact_registry_repository.cement_plant_repo.name
}
