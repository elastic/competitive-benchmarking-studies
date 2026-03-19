variable "project_id" {
  type        = string
}

provider "google" {
  project = var.project_id
  region  = "us-central"
  credentials = file("credentials.json")
}
