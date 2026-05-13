resource "google_container_cluster" "benchmark" {
  name                = var.cluster_name
  location            = var.location
  deletion_protection = var.deletion_protection

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "main_workers" {
  name       = var.main_pool_name
  cluster    = google_container_cluster.benchmark.id
  node_count = var.main_pool_node_count

  node_config {
    machine_type = var.main_pool_machine_type
    disk_type    = var.main_pool_disk_type
    disk_size_gb = var.main_pool_disk_size_gb
  }
}

resource "google_container_node_pool" "jingra" {
  name       = var.jingra_pool_name
  cluster    = google_container_cluster.benchmark.id
  node_count = var.jingra_pool_node_count

  node_config {
    machine_type = var.jingra_pool_machine_type
    disk_size_gb = var.jingra_pool_disk_size_gb
  }
}

resource "google_container_node_pool" "ui" {
  count = var.enable_ui_node_pool ? 1 : 0

  name       = var.ui_pool_name
  cluster    = google_container_cluster.benchmark.id
  node_count = var.ui_pool_node_count

  node_config {
    machine_type = var.ui_pool_machine_type
    disk_size_gb = var.ui_pool_disk_size_gb
  }
}
