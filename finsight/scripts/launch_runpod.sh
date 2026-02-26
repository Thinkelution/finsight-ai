#!/bin/bash
set -e

# FinSight AI — Launch RunPod training instance
# Uses RunPod REST API to create a GPU pod and start training

RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: Set RUNPOD_API_KEY environment variable"
    echo "  export RUNPOD_API_KEY=rpa_..."
    exit 1
fi

API_URL="https://rest.runpod.io/v1/pods"

echo "=== FinSight AI — Creating RunPod Training Instance ==="
echo ""

# Create pod with RTX 4090, PyTorch image, persistent volume
RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL" \
  --request POST \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --data '{
    "name": "finsight-training",
    "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
    "cloudType": "COMMUNITY",
    "containerDiskInGb": 100,
    "volumeInGb": 50,
    "volumeMountPath": "/workspace",
    "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "ports": "22/tcp,8888/http",
    "startSsh": true,
    "env": []
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

echo "HTTP Status: $HTTP_CODE"

if [ "$HTTP_CODE" != "200" ] && [ "$HTTP_CODE" != "201" ]; then
    echo "ERROR: Failed to create pod"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    exit 1
fi

POD_ID=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
echo ""
echo "Pod created successfully!"
echo "  Pod ID: $POD_ID"
echo ""
echo "Monitor at: https://www.runpod.io/console/pods"
echo ""
echo "Once the pod is running, SSH in and run:"
echo "  1. git clone https://github.com/Thinkelution/finsight-ai.git /workspace/finsight-ai"
echo "  2. cd /workspace/finsight-ai"
echo "  3. python finsight/training/runpod_train.py"
echo ""
echo "Or copy the training script directly:"
echo "  scp finsight/training/runpod_train.py root@<pod-ip>:/workspace/"
echo "  ssh root@<pod-ip> 'cd /workspace && python runpod_train.py'"
