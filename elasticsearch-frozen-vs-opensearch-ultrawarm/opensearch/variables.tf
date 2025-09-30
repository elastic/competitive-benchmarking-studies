variable "opensearch_domain_name" {
  type    = string
  default = "benchmarking-opensearch"
}

variable "opensearch_master_user_password" {
  type      = string
  sensitive = true
}

variable "my_public_ip" {
  type = string
}
