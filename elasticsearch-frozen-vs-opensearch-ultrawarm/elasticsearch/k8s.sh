#!/usr/bin/env bash

CRD_EXISTS=$(kubectl get crd elasticsearches.elasticsearch.k8s.elastic.co --ignore-not-found)

if [ -z "$CRD_EXISTS" ]; then
  echo "ECK CRDs not found. Installing..."
  kubectl create -f https://download.elastic.co/downloads/eck/3.1.0/crds.yaml
fi

OPERATOR_EXISTS=$(kubectl get po -n elastic-system --no-headers | grep '^elastic-operator')

if [ -z "$OPERATOR_EXISTS" ]; then
  echo "ECK Operator not found. Installing..."
  kubectl apply -f https://download.elastic.co/downloads/eck/3.1.0/operator.yaml
fi

EBS_CSI_DRIVER_EXISTS=$(kubectl get pods -n kube-system | grep ebs)

if [ -z "$EBS_CSI_DRIVER_EXISTS" ]; then
  helm repo add aws-ebs-csi-driver https://kubernetes-sigs.github.io/aws-ebs-csi-driver
  helm repo update
  helm upgrade --install aws-ebs-csi-driver aws-ebs-csi-driver/aws-ebs-csi-driver \
    --namespace kube-system \
    --set controller.serviceAccount.create=true \
    --set controller.serviceAccount.name=ebs-csi-controller-sa
fi

kubectl apply -f elasticsearch-cluster.yml
