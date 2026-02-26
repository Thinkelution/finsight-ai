#!/bin/bash
set -e

echo '=== FinSight AI - Mac mini Setup ==='

# Install Homebrew if needed
which brew || /bin/bash -c '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)'

# Install system tools
brew install python@3.11 redis git

# Install Docker Desktop (for Qdrant & Grafana)
brew install --cask docker

# Install Ollama
brew install ollama

# Pull embedding model
ollama pull nomic-embed-text

# Pull base model (before fine-tuning available)
ollama pull qwen2.5:14b

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Install playwright browsers
playwright install chromium

# Start infrastructure
docker compose -f finsight/docker/docker-compose.yml up -d

echo '=== Setup complete ==='
echo 'Activate venv: source venv/bin/activate'
echo 'Start API:     uvicorn finsight.api.main:app --host 0.0.0.0 --port 8000'
echo 'Start workers: celery -A finsight.workers.celery_app worker --loglevel=info'
