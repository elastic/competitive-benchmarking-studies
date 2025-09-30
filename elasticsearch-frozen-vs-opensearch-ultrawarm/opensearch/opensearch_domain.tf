data "aws_caller_identity" "current" {}

resource "aws_opensearch_domain" "benchmarking_opensearch" {
  domain_name    = var.opensearch_domain_name
  engine_version = "OpenSearch_3.1"

  cluster_config {
    instance_type  = "m5.4xlarge.search"
    instance_count = 3

    # dedicated_master_enabled = true
    # dedicated_master_type    = "m5.large.search"
    # dedicated_master_count   = 3

    # warm_enabled = true
    # warm_count   = 3
    # warm_type    = "ultrawarm1.medium.search"
  }

  encrypt_at_rest {
    enabled = true
  }

  node_to_node_encryption {
    enabled = true
  }

  advanced_security_options {
    enabled                        = true
    internal_user_database_enabled = true
    master_user_options {
      master_user_name     = "admin"
      master_user_password = var.opensearch_master_user_password
    }
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 300
    volume_type = "gp3"
  }

  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }

  access_policies = <<POLICY
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": "*",
          "Action": "es:*",
          "Resource": "arn:aws:es:us-east-1:${data.aws_caller_identity.current.account_id}:domain/${var.opensearch_domain_name}/*",
          "Condition": {
            "IpAddress": { "aws:SourceIp": "${var.my_public_ip}" }
          }
        }
      ]
    }
    POLICY
}
