"""
API endpoint testleri.
Çalıştırmak için: pytest src/test_api.py -v
API'nin 8000'de çalışıyor olması gerekir.
"""
import pytest
import requests

BASE_URL = "http://localhost:8000"

# ── Health ────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_root():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200

# ── Predict ───────────────────────────────────────────────────

def test_predict_positive():
    r = requests.post(f"{BASE_URL}/predict", json={
        "text": "Revenue surged 40% driven by record-breaking demand."
    })
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] == "positive"
    assert data["confidence"] > 0.8
    assert data["language"] == "en"
    assert "keywords" in data
    assert "risk_score" in data
    assert "risk_level" in data

def test_predict_negative():
    r = requests.post(f"{BASE_URL}/predict", json={
        "text": "Stock prices crashed after company reported massive losses."
    })
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] == "negative"
    assert data["confidence"] > 0.7

def test_predict_neutral():
    r = requests.post(f"{BASE_URL}/predict", json={
        "text": "Net sales remained stable compared to the previous fiscal year."
    })
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] == "neutral"

def test_predict_turkish():
    r = requests.post(f"{BASE_URL}/predict", json={
        "text": "BIST 100 rekor kırdı, yatırımcılar büyük kazanç elde etti."
    })
    assert r.status_code == 200
    data = r.json()
    assert data["language"] == "tr"
    assert data["translated_text"] is not None
    assert data["sentiment"] in ["positive", "neutral", "negative"]

def test_predict_response_schema():
    r = requests.post(f"{BASE_URL}/predict", json={
        "text": "Company earnings beat expectations."
    })
    assert r.status_code == 200
    data = r.json()
    required_fields = [
        "text", "sentiment", "confidence", "language",
        "scores", "keywords", "risk_score", "risk_level", "latency_ms"
    ]
    for field in required_fields:
        assert field in data, f"Eksik alan: {field}"
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert 0 <= data["confidence"] <= 1
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

# ── Validation ────────────────────────────────────────────────

def test_predict_empty_text():
    r = requests.post(f"{BASE_URL}/predict", json={"text": ""})
    assert r.status_code == 422

def test_predict_too_long_text():
    r = requests.post(f"{BASE_URL}/predict", json={"text": "x" * 2001})
    assert r.status_code == 422

# ── Batch ─────────────────────────────────────────────────────

def test_batch_predict():
    r = requests.post(f"{BASE_URL}/predict/batch", json={
        "texts": [
            "Profits rose sharply this quarter.",
            "Company filed for bankruptcy.",
            "Sales remained flat year over year.",
        ]
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    assert data["results"][0]["sentiment"] == "positive"
    assert data["results"][1]["sentiment"] == "negative"
    assert data["results"][2]["sentiment"] == "neutral"

def test_batch_too_large():
    r = requests.post(f"{BASE_URL}/predict/batch", json={
        "texts": ["text"] * 33
    })
    assert r.status_code == 422

def test_batch_empty():
    r = requests.post(f"{BASE_URL}/predict/batch", json={"texts": []})
    assert r.status_code == 422

# ── Monitoring ────────────────────────────────────────────────

def test_monitoring_stats():
    r = requests.get(f"{BASE_URL}/monitoring/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
