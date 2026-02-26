#!/bin/bash
set -e

echo '=== FinSight AI - RunPod Training Setup ==='

apt-get update && apt-get install -y git wget curl

pip install --upgrade pip
pip install unsloth[colab-new] trl peft datasets bitsandbytes
pip install transformers accelerate sentencepiece

# Clone llama.cpp for GGUF conversion
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make -j$(nproc)
    pip install -r requirements.txt
    cd ..
fi

echo '=== RunPod setup complete ==='
echo 'Train:    python -m finsight.training.train_lora'
echo 'Merge:    python -m finsight.training.merge_and_quantize'
echo 'Evaluate: python -m finsight.training.evaluate'
