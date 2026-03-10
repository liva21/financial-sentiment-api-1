# 📊 Financial Sentiment Analysis API

> Türkçe ve İngilizce finansal haber metinlerini **positive / neutral / negative** olarak sınıflandıran, production-ready NLP servisi.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)

---

## 🏗️ Mimari
```
┌─────────────────────────────────────────────────────┐
│                   DATA SOURCES                      │
│         CNBC Finance RSS · Hürriyet Ekonomi RSS     │
└──────────────────┬──────────────────────────────────┘
                   │ feedparser + BeautifulSoup
┌──────────────────▼──────────────────────────────────┐
│              NEWS COLLECTOR                         │
│         Her 30 dakikada otomatik güncelleme         │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP POST /predict
┌──────────────────▼──────────────────────────────────┐
│                FastAPI (port 8000)                  │
│                                                     │
│   LANGUAGE ROUTER                                   │
│   Türkçe → Helsinki-NLP/opus-mt-tr-en → çeviri     │
│   İngilizce → ProsusAI/FinBERT (fine-tuned)        │
│                                                     │
│   ENRICHMENT LAYER                                  │
│   Keywords · Risk Score · Risk Level                │
│                                                     │
│   POST /predict         tek metin analizi           │
│   POST /predict/batch   toplu analiz (max 32)       │
│   GET  /monitoring/stats dashboard verileri         │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│              SQLite Database                        │
│      requests tablosu · news tablosu                │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│           Streamlit Demo (port 8501)                │
│   🔍 Tek Tahmin · 📋 Batch · 📊 Monitoring          │
│   💡 Örnekler  · 📰 Canlı Haberler                  │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
git clone https://github.com/USERNAME/financial-sentiment-api.git
cd financial-sentiment-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Modeli Eğit
```bash
python src/train.py
```

### 3. API'yi Başlat
```bash
uvicorn src.api:app --port 8000
```

### 4. Demo'yu Başlat
```bash
PYTHONPATH=. streamlit run src/demo.py
```

### 5. Docker ile Çalıştır
```bash
docker-compose up --build
```

---

## 📡 API Kullanımı

### Tek Tahmin
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Revenue surged 40% driven by record-breaking demand."}'
```

**Response:**
```json
{
  "text": "Revenue surged 40%...",
  "translated_text": null,
  "sentiment": "positive",
  "confidence": 0.9806,
  "language": "en",
  "scores": {
    "negative": 0.0159,
    "neutral": 0.0035,
    "positive": 0.9806
  },
  "keywords": ["surged", "record", "demand"],
  "risk_score": 0.15,
  "risk_level": "LOW",
  "latency_ms": 312.4
}
```

### Türkçe Metin
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "BIST 100 rekor kırdı, yatırımcılar büyük kazanç elde etti."}'
```

**Response:**
```json
{
  "text": "BIST 100 rekor kırdı...",
  "translated_text": "BIST broke 100 records, investors made a huge profit.",
  "sentiment": "positive",
  "confidence": 0.9529,
  "language": "tr",
  ...
}
```

### Batch Tahmin
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Profits surged 40% this quarter.",
      "Company declared bankruptcy amid mounting losses.",
      "Net sales remained stable year over year."
    ]
  }'
```

---

## 🧠 Model

| Özellik | Detay |
|---|---|
| Base Model | ProsusAI/FinBERT |
| Dataset | financial_phrasebank (sentences_allagree) |
| Train/Val/Test | 80% / 10% / 10% |
| Test Accuracy | 0.978 |
| Test F1 Macro | 0.963 |
| Labels | negative · neutral · positive |
| Çeviri | Helsinki-NLP/opus-mt-tr-en |

---

## 📁 Proje Yapısı
```
financial-sentiment-api/
├── src/
│   ├── api.py              # FastAPI endpoints
│   ├── multilingual.py     # Dil tespiti + çeviri pipeline
│   ├── enrichment.py       # Keyword extraction + risk skoru
│   ├── database.py         # SQLite monitoring
│   ├── news_collector.py   # RSS feed collector
│   ├── train.py            # Model fine-tuning
│   ├── evaluate.py         # Model değerlendirme
│   └── demo.py             # Streamlit arayüzü
├── models/
│   └── finbert-finetuned/  # Eğitilmiş model
├── data/
│   ├── financial_phrasebank.csv
│   ├── test_set.csv
│   └── monitoring.db
├── notebooks/
│   └── 01_eda.ipynb
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
└── requirements.txt
```

---

## ⚠️ Bilinen Limitasyonlar

- **Türkçe çeviri kalitesi:** Helsinki-NLP modeli genel amaçlı, finansal terminolojide zaman zaman yetersiz kalıyor. Daha iyi sonuç için finansal domaine özel çeviri modeli veya Türkçe FinBERT gerekli.
- **Rule-based keyword extraction:** RAKE veya KeyBERT ile geliştirilebilir.
- **Latency:** Çeviri pipeline'ı nedeniyle Türkçe metinler ~2-3 saniye sürüyor. Model quantization ile iyileştirilebilir.
- **RSS erişimi:** Bazı Türk haber siteleri bot engellemesi uyguluyor.

---

## 🛠️ Teknoloji Stack

| Katman | Teknoloji |
|---|---|
| Model | HuggingFace Transformers, PyTorch |
| Serving | FastAPI, Uvicorn |
| Çeviri | Helsinki-NLP/opus-mt-tr-en |
| Demo | Streamlit, Plotly |
| Storage | SQLite |
| Container | Docker, Docker Compose |
| Data | HuggingFace Datasets |

---

## 📊 Model Performansı
```
Test Seti (227 örnek):

              precision  recall  f1-score  support
    negative     0.97     0.95     0.96      30
     neutral     0.99     0.99     0.99     139
    positive     0.96     0.98     0.97      58

    accuracy                         0.978    227
   macro avg     0.974    0.973     0.963    227
```

---

## ‍💻 Geliştirici

**Liva Nur  Karanfil**


