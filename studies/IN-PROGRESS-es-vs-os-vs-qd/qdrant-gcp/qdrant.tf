resource "google_container_cluster" "qdrant_context_engineering_benchmark" {
  name                = "qdrant-context-engineering-benchmark"
  location            = "us-central1-a"
  deletion_protection = false

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "qdrant_context_engineering_nodes" {
  name       = "qdrant-context-engineering-nodepool"
  cluster    = google_container_cluster.qdrant_context_engineering_benchmark.id
  node_count = 3

  node_config {
    machine_type = "e2-standard-4"
    disk_size_gb = 100
  }
}

resource "google_container_node_pool" "jingra_node" {
  name       = "jingra-nodepool"
  cluster    = google_container_cluster.qdrant_context_engineering_benchmark.id
  node_count = 2

  node_config {
    machine_type = "e2-standard-4"
    disk_size_gb = 50
  }
}
