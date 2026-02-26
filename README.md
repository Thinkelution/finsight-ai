# FinSight AI

Real-time financial intelligence system that continuously ingests live news feeds, RSS sources, and market data to provide up-to-the-minute analysis of global financial markets.

## Architecture

- **Ingestion Layer** — RSS feeds, web scraping, live market data (yfinance), social signals
- **Processing Pipeline** — NER (spaCy), sentiment (FinBERT), chunking, embedding (nomic-embed-text)
- **Vector Storage** — Qdrant with time-decay scoring and metadata filtering
- **Inference** — RAG query engine with Qwen 2.5 14B via Ollama
- **API** — FastAPI with streaming responses
- **Workers** — Celery + Redis for async ingestion
- **Training** — LoRA fine-tuning on RunPod GPU instances

## Coverage

- **Forex** — major, minor, and exotic currency pairs
- **Equities** — US, European, and Asian markets
- **Commodities** — oil, gold, silver, agricultural
- **Macroeconomics** — central bank decisions, CPI, GDP, employment data
- **Crypto** — BTC, ETH, and major altcoins (optional)
- **Sentiment** — social signals from Reddit, StockTwits

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- Ollama

### Setup

```bash
# Run one-shot setup (installs everything)
chmod +x finsight/scripts/setup_mac.sh
./finsight/scripts/setup_mac.sh

# Or manual setup:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start infrastructure
docker compose -f finsight/docker/docker-compose.yml up -d

# Pull embedding model
ollama pull nomic-embed-text
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
# Start API server
uvicorn finsight.api.main:app --host 0.0.0.0 --port 8000

# Start ingestion workers (separate terminal)
celery -A finsight.workers.celery_app worker --loglevel=info

# Start periodic task scheduler (separate terminal)
celery -A finsight.workers.celery_app beat --loglevel=info
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Why did the dollar strengthen today?"}'
```

## Project Structure

```
finsight/
├── ingestion/          # RSS, web scraping, market data, social feeds
├── processing/         # NER, sentiment, chunking, embedding
├── storage/            # Qdrant client, indexer, retriever, summariser
├── inference/          # RAG query engine, context builder, alerter
├── api/                # FastAPI server and routes
├── training/           # RunPod LoRA training pipeline
├── workers/            # Celery async tasks
├── config/             # Settings and logging
├── docker/             # Docker Compose + Prometheus config
├── scripts/            # Setup and deployment scripts
└── tests/              # Test suite
```

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/finsight)

## Training (RunPod)

See `finsight/scripts/setup_runpod.sh` for GPU instance setup. Training uses Unsloth + QLoRA on Qwen 2.5 14B.

## License

MIT — see LICENSE
