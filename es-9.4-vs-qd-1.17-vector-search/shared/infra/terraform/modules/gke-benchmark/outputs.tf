output "cluster_id" {
  value = google_container_cluster.benchmark.id
}

output "cluster_name" {
  value = google_container_cluster.benchmark.name
}

output "cluster_endpoint" {
  value     = google_container_cluster.benchmark.endpoint
  sensitive = true
}
