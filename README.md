# FinSight AI

Real-time financial intelligence system that continuously ingests live news feeds, RSS sources, and market data to provide up-to-the-minute analysis of global financial markets.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                           │
│  RSS Feeds (12+)  │  Web Scraper  │  yfinance  │  Social    │
│  Reuters, CNBC,   │  Bloomberg,   │  Forex,    │  Reddit,   │
│  FXStreet, FT     │  SEC EDGAR    │  Indices   │  StockTwits│
└──────────────────────┬───────────────────────────────────────┘
                       │ Redis Streams + Celery
┌──────────────────────▼───────────────────────────────────────┐
│                 PROCESSING PIPELINE                          │
│  1. Clean & deduplicate (SHA256)                             │
│  2. Entity extraction (spaCy NER + regex)                    │
│  3. Sentiment scoring (FinBERT)                              │
│  4. Chunk (500 tokens, 50 overlap)                           │
│  5. Embed (nomic-embed-text, 768d)                           │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│              VECTOR DATABASE (Qdrant)                         │
│  • Time-decay scoring (7hr half-life)                        │
│  • Filter by asset class, source, time window                │
│  • Auto-expiry: 7-day retention                              │
│  • Rolling 24hr market summary (refreshed every 30 min)      │
└──────────────────────┬───────────────────────────────────────┘
                       │ at query time
┌──────────────────────▼───────────────────────────────────────┐
│                LLM INFERENCE LAYER                           │
│  Qwen 2.5 14B Q4 (fine-tuned via LoRA on RunPod)            │
│  Groq fallback for high availability                         │
│  Context: top-k chunks + live prices + 24hr summary          │
└──────────────────────────────────────────────────────────────┘
```

## Coverage

- **Forex** — major, minor, and exotic currency pairs
- **Equities** — US, European, and Asian markets
- **Commodities** — oil, gold, silver, agricultural
- **Macroeconomics** — central bank decisions, CPI, GDP, employment data
- **Crypto** — BTC, ETH, and major altcoins
- **Sentiment** — social signals from Reddit, StockTwits

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- Ollama

### Setup

```bash
# One-shot setup (installs everything)
chmod +x finsight/scripts/setup_mac.sh
./finsight/scripts/setup_mac.sh

# Or manual setup:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
playwright install chromium

# Start infrastructure
docker compose -f finsight/docker/docker-compose.yml up -d

# Pull models
ollama pull nomic-embed-text
ollama pull qwen2.5:14b
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
# Simple query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Why did the dollar strengthen today?"}'

# With asset class filter
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What moved gold today?", "asset_class": "commodities"}'

# Multi-turn conversation
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Follow up on EUR/USD", "session_id": "my-session-123"}'

# Live market prices
curl http://localhost:8000/market/live

# Health check
curl http://localhost:8000/health

# Recent alerts
curl http://localhost:8000/alerts
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Financial question answering (RAG) |
| GET | `/market/live` | All live prices |
| GET | `/market/forex` | Forex rates |
| GET | `/market/indices` | Index levels |
| GET | `/market/commodities` | Commodity prices |
| GET | `/market/crypto` | Crypto prices |
| GET | `/market/history/{symbol}` | Intraday price history |
| GET | `/health` | Service health check |
| GET | `/alerts` | Recent system alerts |
| GET | `/metrics` | Prometheus metrics |

## Project Structure

```
finsight/
├── ingestion/          # RSS, web scraping, market data, social feeds
│   ├── rss_fetcher.py      # Polls 12+ financial news RSS feeds
│   ├── web_scraper.py      # Scrapes news sites, SEC EDGAR, PR wires
│   ├── social_fetcher.py   # Reddit and StockTwits
│   ├── market_data.py      # yfinance live prices
│   ├── deduplicator.py     # SHA256 Redis-backed dedup
│   └── sources.yaml        # All feed URLs and config
├── processing/         # NER, sentiment, chunking, embedding
│   ├── cleaner.py          # HTML strip, boilerplate removal
│   ├── chunker.py          # Sliding window (500 tokens, 50 overlap)
│   ├── embedder.py         # nomic-embed-text via Ollama
│   ├── ner.py              # spaCy + regex entity extraction
│   ├── sentiment.py        # FinBERT with keyword fallback
│   └── pipeline.py         # Orchestrates full processing flow
├── storage/            # Qdrant client, indexer, retriever, summariser
│   ├── qdrant_store.py     # Connection + collection management
│   ├── indexer.py          # Batch chunk insertion
│   ├── retriever.py        # Time-weighted similarity search
│   └── summariser.py       # Rolling 24h market narrative
├── inference/          # RAG query engine, context builder, alerter
│   ├── query_engine.py     # Full RAG pipeline
│   ├── context_builder.py  # Assembles prices + chunks + summary
│   ├── prompt_templates.py # System and user prompts
│   ├── alerter.py          # Price spikes, breaking news, correlations
│   ├── fallback.py         # Groq API fallback
│   └── chat_history.py     # Multi-turn conversation
├── api/                # FastAPI server and routes
│   ├── main.py             # App entry point with Prometheus
│   ├── schemas.py          # Pydantic models
│   ├── rate_limiter.py     # Redis-backed rate limiting
│   ├── metrics.py          # Custom Prometheus metrics
│   └── routes/             # /query, /market, /health, /alerts
├── training/           # RunPod LoRA training pipeline
│   ├── prepare_dataset.py  # HuggingFace + manual dataset prep
│   ├── train_lora.py       # Unsloth QLoRA on Qwen 2.5 14B
│   ├── merge_and_quantize.py  # Merge + GGUF conversion
│   └── evaluate.py         # Benchmark evaluation harness
├── workers/            # Celery async tasks
│   ├── celery_app.py       # App config + beat schedule
│   └── tasks.py            # All background tasks
├── config/             # Settings and logging
├── docker/             # Docker Compose + Prometheus + Grafana
├── scripts/            # Setup and deployment scripts
└── tests/              # Test suite
```

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / finsight)
- **Metrics**: http://localhost:8000/metrics

Dashboard panels: query rate, latency percentiles, ingestion by source, collection size, sentiment distribution, LLM latency, alerts, errors.

## Fine-Tuning (RunPod)

```bash
# 1. Prepare training data
python -m finsight.training.prepare_dataset

# 2. Set up RunPod (on GPU instance)
bash finsight/scripts/setup_runpod.sh

# 3. Train LoRA adapter
python -m finsight.training.train_lora

# 4. Merge and quantize
python -m finsight.training.merge_and_quantize

# 5. Transfer GGUF to Mac and register
scp finsight_qwen14b_q4.gguf user@mac:~/models/
./finsight/scripts/pull_model.sh ~/models/finsight_qwen14b_q4.gguf

# 6. Evaluate
python -m finsight.training.evaluate
python -m finsight.training.evaluate compare  # base vs fine-tuned
```

## Memory Budget (16GB Mac mini M4)

| Component | RAM | Notes |
|-----------|-----|-------|
| macOS + system | ~3 GB | Baseline |
| Qwen 2.5 14B Q4 | ~8-9 GB | Unified memory |
| nomic-embed-text | ~0.5 GB | Small, fast |
| Qdrant | ~0.5-1 GB | Depends on index |
| Redis | ~0.1 GB | Lightweight |
| Python pipeline | ~0.5-1 GB | FastAPI + Celery |
| **Total** | **~13-14 GB** | Fits with headroom |

## License

MIT — see LICENSE
