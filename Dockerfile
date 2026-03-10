# ── Base image ─────────────────────────────────────────────
FROM python:3.11-slim

# ── Sistem bağımlılıkları ───────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Çalışma dizini ──────────────────────────────────────────
WORKDIR /app

# ── Python bağımlılıkları ───────────────────────────────────
# Önce sadece requirements.txt kopyala — layer cache'i korur
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# ── Uygulama dosyaları ──────────────────────────────────────
COPY src/      ./src/
# Model dosyaları imaja gömülmez — docker run -v ile mount edilir.

# ── Ortam değişkenleri ──────────────────────────────────────
ENV MODEL_DIR=/app/models/finbert-finetuned \
    MAX_LENGTH=128 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Port ────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Başlangıç komutu ────────────────────────────────────────
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
