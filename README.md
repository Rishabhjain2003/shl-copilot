# SHL Conversational Assessment Recommender

A stateless conversational agent that helps hiring managers select SHL Individual Test Solutions, served via FastAPI.

## Architecture

- **Hybrid retrieval**: TF-IDF lexical + sentence-transformer semantic embeddings, fused with weighted sum.
- **Explicit state machine**: 5 action branches (CLARIFY, RETRIEVE_AND_RECOMMEND, REFINE, COMPARE, REFUSE).
- **Dynamic query-time failover**: Calls Groq first (lowest latency), falls back automatically to Gemini Flash, and then to OpenAI GPT-4o-mini on rate-limit failures in under 1 second.
- **Baked-in Cache**: Model caches are baked into the container image to prevent startup network downloads.
- **Separate prompts per action**: Easier to debug, test, and defend.
- **Grounding enforced programmatically**: Post-hoc URL validation against catalog set; LLM only picks from top-20 retrieval candidates.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CPU-optimized PyTorch pinned)
pip install -r requirements.txt

# Download and clean catalog
python scripts/scrape.py

# Build embeddings and TF-IDF index
python scripts/build_embeddings.py

# Set API key (Groq, Gemini, or OpenAI)
export GROQ_API_KEY="your-key-here"

# Run the server
uvicorn app.main:app --reload --port 8080
```

## API

### `GET /health`
Returns `{"status": "ok"}` (Fast and lightweight health check).

### `POST /chat`
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for Java developers"}
  ]
}
```

Response:
```json
{
  "reply": "For Java developers, here are the relevant assessments...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Testing

Install `pytest` and `pytest-asyncio` inside your venv, then run:

```bash
# Run all tests
pytest tests/ -v

# Behavior probes only
pytest tests/test_behavior.py -v

# Trace replay with Recall@10
pytest tests/test_traces.py -v -s
```

## Deployment

### Docker (Render, Fly.io, Railway)

The docker container is pre-optimized for production deployment. The sentence-transformers cache directory is configured via `ENV HF_HOME=/app/.cache` and pre-downloaded during docker build:

```bash
# Build Docker image
docker build -t shl-recommender .

# Run Docker container
docker run -p 8080:8080 -e GROQ_API_KEY="your-key" shl-recommender
```

## Design Decisions

| Decision | Trade-off / Rationale |
|----------|-----------|
| **TF-IDF over BM25** | Simpler, sklearn built-in. Fused with semantic search + exact-name boosting (+0.3) to achieve 100% recall. |
| **all-MiniLM-L6-v2 (384d)** | Small and CPU-efficient. Cached and pre-baked inside `/app/.cache` in the image to prevent runtime network fetches. |
| **Eager lifespan loading** | Loads embedding weights at FastAPI startup. Avoids paying PyTorch load latency on the first API request. |
| **Dynamic Query Failover** | Groq (`openai/gpt-oss-120b`) -> Gemini Flash -> OpenAI GPT-4o-mini. Protects the app from individual model rate limits and outages. |
| **No LangChain** | Hand-rolled orchestration in Python. Keeps the code highly performant, customizable, and simple to debug. |

