resource "google_container_cluster" "os_context_engineering_benchmark" {
  name                = "os-context-engineering-benchmark"
  location            = "us-central1-a"
  deletion_protection = false

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "os_context_engineering_nodes" {
  name       = "os-context-engineering-nodepool"
  cluster    = google_container_cluster.os_context_engineering_benchmark.id
  node_count = 3

  node_config {
    machine_type = "e2-standard-4"
    disk_size_gb = 100
  }
}

resource "google_container_node_pool" "osd_context_engineering_node" {
  name       = "osd-context-engineering-nodepool"
  cluster    = google_container_cluster.os_context_engineering_benchmark.id
  node_count = 1

  node_config {
    machine_type = "e2-medium"
    disk_size_gb = 50
  }
}
