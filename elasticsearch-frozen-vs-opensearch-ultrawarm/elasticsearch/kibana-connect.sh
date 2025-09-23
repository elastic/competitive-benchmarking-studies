#!/usr/bin/env bash

aws eks update-kubeconfig --region us-east-1 --name elasticsearch-benchmarking-cluster --profile ecdev

NAMESPACE="default"

ELASTIC_PASSWORD=$(kubectl get secret es-cluster-es-elastic-user -n $NAMESPACE -o go-template='{{.data.elastic | base64decode}}{{"\n"}}')

echo "Elastic password: $ELASTIC_PASSWORD"

kubectl port-forward svc/es-cluster-kb-http 5601
