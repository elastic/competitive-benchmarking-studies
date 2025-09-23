data "aws_iam_policy_document" "benchmarking_eks_cluster_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "benchmarking_eks_cluster_role" {
  name               = "eksClusterRole"
  assume_role_policy = data.aws_iam_policy_document.benchmarking_eks_cluster_assume_role.json
}

resource "aws_iam_role_policy_attachment" "benchmarking_eks_cluster_AmazonEKSClusterPolicy" {
  role       = aws_iam_role.benchmarking_eks_cluster_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

data "aws_iam_policy_document" "benchmarking_eks_node_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "benchmarking_eks_node_role" {
  name               = "eksNodeRole"
  assume_role_policy = data.aws_iam_policy_document.benchmarking_eks_node_assume_role.json
}

resource "aws_iam_role_policy_attachment" "benchmarking_eks_node_AmazonEKSWorkerNodePolicy" {
  role       = aws_iam_role.benchmarking_eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "benchmarking_eks_node_AmazonEC2ContainerRegistryReadOnly" {
  role       = aws_iam_role.benchmarking_eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "benchmarking_eks_node_AmazonEKS_CNI_Policy" {
  role       = aws_iam_role.benchmarking_eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "benchmarking_eks_node_AmazonEBSCSIDriverPolicy" {
  role       = aws_iam_role.benchmarking_eks_node_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

resource "aws_iam_policy" "benchmarking_eks_ebs_csi_driver_policy" {
  name        = "AmazonEBSCSIDriverPolicy"
  description = "Policy for EBS CSI Driver"
  policy      = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:CreateSnapshot",
                "ec2:AttachVolume",
                "ec2:DetachVolume",
                "ec2:ModifyVolume",
                "ec2:DescribeAvailabilityZones",
                "ec2:DescribeInstances",
                "ec2:DescribeSnapshots",
                "ec2:DescribeTags",
                "ec2:DescribeVolumes",
                "ec2:DescribeVolumesModifications",
                "ec2:CreateTags",
                "ec2:DeleteTags",
                "ec2:CreateVolume",
                "ec2:DeleteVolume"
            ],
            "Resource": "*"
        }
    ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "benchmarking_eks_node_attach_ebs_csi_policy" {
  role       = aws_iam_role.benchmarking_eks_node_role.name
  policy_arn = aws_iam_policy.benchmarking_eks_ebs_csi_driver_policy.arn
}
