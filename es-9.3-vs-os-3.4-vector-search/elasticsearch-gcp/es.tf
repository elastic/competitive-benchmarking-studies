resource "google_container_cluster" "es_context_engineering_benchmark" {
  name                = "es-context-engineering-benchmark"
  location            = "us-central1-a"
  deletion_protection = false

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "es_context_engineering_nodes" {
  name       = "es-context-engineering-nodepool"
  cluster    = google_container_cluster.es_context_engineering_benchmark.id
  node_count = 3

  node_config {
    machine_type = "e2-standard-2"
    disk_size_gb = 50
  }
}

resource "google_container_node_pool" "kb_context_engineering_node" {
  name       = "kb-context-engineering-nodepool"
  cluster    = google_container_cluster.es_context_engineering_benchmark.id
  node_count = 1

  node_config {
    machine_type = "e2-medium"
    disk_size_gb = 12
  }
}
