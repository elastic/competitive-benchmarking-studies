resource "google_container_cluster" "os_context_engineering_benchmark" {
  name                = "os-context-engineering-benchmark-large"
  location            = "us-central1-a"
  deletion_protection = false

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "os_context_engineering_nodes" {
  name       = "os-context-engineering-nodepool-large"
  cluster    = google_container_cluster.os_context_engineering_benchmark.id
  node_count = 6

  node_config {
    machine_type = "e2-standard-16"
    disk_size_gb = 200
  }
}

resource "google_container_node_pool" "osd_context_engineering_node" {
  name       = "osd-context-engineering-nodepool-large"
  cluster    = google_container_cluster.os_context_engineering_benchmark.id
  node_count = 1

  node_config {
    machine_type = "e2-medium"
    disk_size_gb = 120
  }
}
