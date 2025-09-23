resource "aws_eks_cluster" "elasticsearch_cluster" {
  name     = "elasticsearch-benchmarking"
  role_arn = aws_iam_role.benchmarking_eks_cluster_role.arn
  version  = "1.29"

  vpc_config {
    subnet_ids = [aws_subnet.benchmarking_elasticsearch_subnet_public1.id, aws_subnet.benchmarking_elasticsearch_subnet_public2.id]
  }
}

resource "aws_eks_node_group" "elasticsearch_data_nodes" {
  cluster_name    = aws_eks_cluster.elasticsearch_cluster.name
  node_group_name = "elasticsearch-nodes"
  node_role_arn   = aws_iam_role.benchmarking_eks_node_role.arn
  subnet_ids      = [aws_subnet.benchmarking_elasticsearch_subnet_public1.id]

  scaling_config {
    desired_size = 3
    min_size     = 3
    max_size     = 3
  }

  instance_types = ["m5.4xlarge"]
  disk_size      = 100
}

resource "aws_eks_node_group" "kibana_node" {
  cluster_name    = aws_eks_cluster.elasticsearch_cluster.name
  node_group_name = "kibana-node"
  node_role_arn   = aws_iam_role.benchmarking_eks_node_role.arn
  subnet_ids      = [aws_subnet.benchmarking_elasticsearch_subnet_public1.id]
  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 1
  }
  instance_types = ["t3.medium"]
  disk_size      = 20
}
