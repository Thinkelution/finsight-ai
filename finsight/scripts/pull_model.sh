#!/bin/bash
set -e

MODEL_PATH="${1:-finsight_qwen14b_q4.gguf}"
MODEL_NAME="${2:-finsight-qwen14b}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Usage: ./pull_model.sh <path_to_gguf> [model_name]"
    exit 1
fi

echo "=== Registering $MODEL_NAME in Ollama ==="

cat > /tmp/Modelfile << 'EOF'
FROM ./PLACEHOLDER_PATH

SYSTEM """You are FinSight, an expert financial markets analyst specialising in forex,
global equities, commodities, and macroeconomics. You have access to real-time news
and live market prices. Always cite your sources. Note the time of the information
you reference. Be precise about numbers and dates. When uncertain, say so explicitly."""

PARAMETER temperature 0.1
PARAMETER num_ctx 8192
PARAMETER stop "### Instruction:"
EOF

sed -i.bak "s|PLACEHOLDER_PATH|$MODEL_PATH|g" /tmp/Modelfile

ollama create "$MODEL_NAME" -f /tmp/Modelfile
rm -f /tmp/Modelfile /tmp/Modelfile.bak

echo "=== Model $MODEL_NAME registered ==="
echo "Test: ollama run $MODEL_NAME"
