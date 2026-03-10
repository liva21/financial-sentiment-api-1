import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)
from langdetect import detect, LangDetectException
import os

FINBERT_DIR     = os.getenv("MODEL_DIR", "models/finbert-finetuned")
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-tr-en"
MAX_LENGTH      = 128

DEVICE_STR = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

_finbert_tokenizer  = None
_finbert_model      = None
_marian_tokenizer   = None
_marian_model       = None

def load_all_models():
    global _finbert_tokenizer, _finbert_model, _marian_tokenizer, _marian_model

    print(f"FinBERT yükleniyor: {FINBERT_DIR}")
    _finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_DIR)
    _finbert_model     = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_DIR
    ).to(DEVICE_STR)
    _finbert_model.eval()
    print(f"  ✓ FinBERT hazır [{DEVICE_STR}]")

    print(f"Çeviri modeli yükleniyor: {TRANSLATE_MODEL}")
    _marian_tokenizer = MarianTokenizer.from_pretrained(TRANSLATE_MODEL)
    _marian_model     = MarianMTModel.from_pretrained(TRANSLATE_MODEL)
    _marian_model.eval()
    print(f"  ✓ Çeviri modeli hazır")

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def translate_to_english(texts: list[str]) -> list[str]:
    inputs  = _marian_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = _marian_model.generate(**inputs)
    return _marian_tokenizer.batch_decode(outputs, skip_special_tokens=True)

def run_finbert(texts: list[str]) -> list[dict]:
    enc = _finbert_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE_STR)

    with torch.no_grad():
        logits = _finbert_model(**enc).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for i, text in enumerate(texts):
        p       = probs[i]
        pred_id = int(np.argmax(p))
        results.append({
            "text"      : text,
            "sentiment" : ID2LABEL[pred_id],
            "confidence": round(float(p[pred_id]), 4),
            "scores"    : {
                "negative": round(float(p[0]), 4),
                "neutral" : round(float(p[1]), 4),
                "positive": round(float(p[2]), 4),
            },
        })
    return results

def run_inference_multilingual(texts: list[str]) -> list[dict]:
    langs   = [detect_language(t) for t in texts]
    results = [None] * len(texts)

    tr_indices = [i for i, l in enumerate(langs) if l == "tr"]
    en_indices = [i for i, l in enumerate(langs) if l != "tr"]

    if tr_indices:
        tr_texts   = [texts[i] for i in tr_indices]
        translated = translate_to_english(tr_texts)
        tr_results = run_finbert(translated)

        for j, idx in enumerate(tr_indices):
            r = tr_results[j]
            results[idx] = {
                "text"           : texts[idx],
                "translated_text": translated[j],
                "sentiment"      : r["sentiment"],
                "confidence"     : r["confidence"],
                "language"       : "tr",
                "scores"         : r["scores"],
            }

    if en_indices:
        en_texts   = [texts[i] for i in en_indices]
        en_results = run_finbert(en_texts)

        for j, idx in enumerate(en_indices):
            r = en_results[j]
            results[idx] = {
                "text"           : texts[idx],
                "translated_text": None,
                "sentiment"      : r["sentiment"],
                "confidence"     : r["confidence"],
                "language"       : "en",
                "scores"         : r["scores"],
            }

    return results
