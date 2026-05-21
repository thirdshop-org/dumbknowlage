FROM python:3.12-slim

WORKDIR /app

# System dependencies for spaCy, audio, OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    tesseract-ocr \
    tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download fr_core_news_lg && \
    python -m spacy download en_core_web_lg

# Copy app code
COPY server/app/ /app/
COPY shared/ /app/shared/

# Data volume
VOLUME ["/app/data"]

# Expose API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=5)"

CMD ["python", "main.py"]
