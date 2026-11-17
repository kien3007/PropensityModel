#!/bin/sh
# Wait for MinIO to become available
until curl -s http://minio:9000/minio/health/live; do
    echo "Waiting for MinIO..."
    sleep 2
done

# Configure client
mc alias set local http://minio:9000 minioadmin minioadmin

# Create Master Dataset & Bronze buckets if not existing
mc mb --ignore-existing local/lakehouse-master
mc mb --ignore-existing local/lakehouse-bronze
mc mb --ignore-existing local/mlflow-artifacts

echo "MinIO buckets initialized successfully."
