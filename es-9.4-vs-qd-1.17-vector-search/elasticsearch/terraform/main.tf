terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

variable "project_id" {
  type = string
}

provider "google" {
  project = var.project_id
  region  = "us-central1"
}

module "gke_benchmark" {
  source = "../../infra/terraform/modules/gke-benchmark"

  cluster_name        = "elasticsearch-benchmark"
  main_pool_name      = "elasticsearch-nodepool"
  enable_ui_node_pool = true
  ui_pool_name        = "kibana-nodepool"

  location                 = var.location
  deletion_protection      = var.deletion_protection
  main_pool_disk_type      = var.main_pool_disk_type
  jingra_pool_name         = var.jingra_pool_name
  main_pool_node_count     = var.main_pool_node_count
  main_pool_machine_type   = var.main_pool_machine_type
  main_pool_disk_size_gb   = var.main_pool_disk_size_gb
  jingra_pool_node_count   = var.jingra_pool_node_count
  jingra_pool_machine_type = var.jingra_pool_machine_type
  jingra_pool_disk_size_gb = var.jingra_pool_disk_size_gb
  ui_pool_node_count       = var.ui_pool_node_count
  ui_pool_machine_type     = var.ui_pool_machine_type
  ui_pool_disk_size_gb     = var.ui_pool_disk_size_gb
}
