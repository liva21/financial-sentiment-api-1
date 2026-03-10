from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional
import time, os
from src.database import init_db, log_request
from src.multilingual import load_all_models, run_inference_multilingual
from src.enrichment import enrich

MODEL_DIR  = os.getenv("MODEL_DIR", "models/finbert-finetuned")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))

app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="Türkçe ve İngilizce finansal metinleri analiz eder.",
    version="3.0.0",
)

@app.on_event("startup")
async def startup():
    init_db()
    load_all_models()

class SentimentRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("text boş olamaz")
        if len(v) > 2000:
            raise ValueError("text 2000 karakteri aşamaz")
        return v

class SentimentScore(BaseModel):
    negative: float
    neutral:  float
    positive: float

class SentimentResponse(BaseModel):
    text:             str
    translated_text:  Optional[str] = None
    sentiment:        str
    confidence:       float
    language:         str
    scores:           SentimentScore
    keywords:         list[str]        # ← YENİ
    risk_score:       float            # ← YENİ
    risk_level:       str              # ← YENİ: LOW / MEDIUM / HIGH
    latency_ms:       float

class BatchRequest(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def validate_batch(cls, v):
        if not v:
            raise ValueError("texts boş olamaz")
        if len(v) > 32:
            raise ValueError("max 32 metin")
        return v

class BatchResponse(BaseModel):
    results:    list[SentimentResponse]
    latency_ms: float

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "version": "3.0.0"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_dir": MODEL_DIR}

@app.get("/monitoring/stats", tags=["Monitoring"])
def monitoring_stats():
    from src.database import get_stats
    return get_stats()

@app.post("/predict", response_model=SentimentResponse, tags=["Inference"])
def predict(req: SentimentRequest):
    t0     = time.perf_counter()
    result = run_inference_multilingual([req.text])[0]

    # Enrichment — İngilizce metin üzerinde çalışır
    analysis_text = result.get("translated_text") or result["text"]
    enriched      = enrich(analysis_text, result["sentiment"], result["confidence"])

    latency = round((time.perf_counter() - t0) * 1000, 2)

    log_request(
        text      = req.text,
        sentiment = result["sentiment"],
        confidence= result["confidence"],
        latency_ms= latency,
        endpoint  = "/predict",
    )

    return SentimentResponse(
        text            = result["text"],
        translated_text = result.get("translated_text"),
        sentiment       = result["sentiment"],
        confidence      = result["confidence"],
        language        = result["language"],
        scores          = SentimentScore(**result["scores"]),
        keywords        = enriched["keywords"],
        risk_score      = enriched["risk_score"],
        risk_level      = enriched["risk_level"],
        latency_ms      = latency,
    )

@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(req: BatchRequest):
    t0      = time.perf_counter()
    results = run_inference_multilingual(req.texts)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    responses = []
    for r in results:
        analysis_text = r.get("translated_text") or r["text"]
        enriched      = enrich(analysis_text, r["sentiment"], r["confidence"])

        log_request(
            text      = r["text"],
            sentiment = r["sentiment"],
            confidence= r["confidence"],
            latency_ms= latency / len(results),
            endpoint  = "/predict/batch",
            batch_size= len(req.texts),
        )

        responses.append(SentimentResponse(
            text            = r["text"],
            translated_text = r.get("translated_text"),
            sentiment       = r["sentiment"],
            confidence      = r["confidence"],
            language        = r["language"],
            scores          = SentimentScore(**r["scores"]),
            keywords        = enriched["keywords"],
            risk_score      = enriched["risk_score"],
            risk_level      = enriched["risk_level"],
            latency_ms      = latency,
        ))

    return BatchResponse(results=responses, latency_ms=latency)
