FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the Hugging Face cache directory to a path inside the workspace
ENV HF_HOME=/app/.cache

# Pre-download the embedding model so it's baked into the image —
# avoids a network fetch from HuggingFace Hub on every cold start.
# Make the cache directory globally readable/writeable for any runtime user.
RUN mkdir -p /app/.cache && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    chmod -R 777 /app/.cache

# Copy application code and data
COPY app/ app/
COPY data/ data/
COPY scripts/ scripts/

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]