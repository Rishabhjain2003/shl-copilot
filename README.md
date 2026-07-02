# SHL Conversational Assessment Recommender

A stateless conversational agent that helps hiring managers select SHL Individual Test Solutions, served via FastAPI.

## Architecture

- **Hybrid retrieval**: TF-IDF lexical + sentence-transformer semantic embeddings, fused with weighted sum
- **Explicit state machine**: 5 action branches (CLARIFY, RETRIEVE_AND_RECOMMEND, REFINE, COMPARE, REFUSE)
- **Separate prompts per action**: Easier to debug, test, and defend
- **Grounding enforced programmatically**: Post-hoc URL validation against catalog set; LLM only picks from top-20 retrieval candidates

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and clean catalog
python scripts/scrape.py

# Build embeddings and TF-IDF index
python scripts/build_embeddings.py

# Set API key
export GEMINI_API_KEY="your-key-here"

# Run the server
uvicorn app.main:app --reload --port 8080
```

## API

### `GET /health`
Returns `{"status": "ok"}`.

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
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Behavior probes only
pytest tests/test_behavior.py -v

# Trace replay with Recall@10
pytest tests/test_traces.py -v -s
```

## Deployment

```bash
# Fly.io
fly deploy

# Or with Docker
docker build -t shl-recommender .
docker run -p 8080:8080 -e GEMINI_API_KEY=your-key shl-recommender
```

## Design Decisions

| Decision | Trade-off |
|----------|-----------|
| TF-IDF over BM25 | Simpler, sklearn built-in. BM25 with term saturation would be better with more time. |
| all-MiniLM-L6-v2 (384d) | Small enough for Docker. all-mpnet-base-v2 (768d) would be higher quality. |
| Gemini Flash | Free-tier friendly, fast. GPT-4o-mini has better structured output adherence. |
| Separate prompts per action | More HTTP calls but easier to debug and unit test individually. |
| No LangChain | Hand-rolled orchestration. Easier to justify, explain, and debug. |
| Precomputed embeddings | No vector DB dependency. Works for 377 items; wouldn't scale past ~10K. |
