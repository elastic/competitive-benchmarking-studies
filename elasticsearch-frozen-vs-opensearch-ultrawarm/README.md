# Elasticsearch Frozen vs OpenSearch UltraWarm

The code in this folder is able to stand up an Elasticsearch cluster in AWS EKS and it can create an OpenSearch domain in AWS.

## Creating the clusters

### `elasticsearch`

```
./terraform.sh
./k8s-connect.sh
./k8s.sh
./kibana-connect.sh
```

### `opensearch`

```
./terraform.sh
```
