data "aws_eks_cluster" "eks" {
  name = aws_eks_cluster.elasticsearch_cluster.name
}

data "aws_eks_cluster_auth" "eks" {
  name = aws_eks_cluster.elasticsearch_cluster.name
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.eks.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.eks.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.eks.token
}
