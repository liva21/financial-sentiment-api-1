---
title: Financial Sentiment API
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Financial Sentiment Analysis API

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://liva21-financial-sentiment.hf.space)
[![Evaluation](https://github.com/liva21/financial-sentiment-api-1/actions/workflows/eval.yml/badge.svg)](https://github.com/liva21/financial-sentiment-api-1/actions/workflows/eval.yml)

An API and web interface for analyzing financial text sentiment using FinBERT.

## Evaluation

Our model is continuously evaluated to ensure high accuracy in identifying financial sentiments, particularly on ambiguous language and financial jargon. 

- **Test Suite:** 30 comprehensive diverse test cases (`pytest`)
- **Pipeline:** Automated evaluation via GitHub Actions
- **Full Report:** See [docs/eval_report.md](docs/eval_report.md) for detailed metrics including Precision, Recall, and F1 per class.

## ATS Use Case Prototype

The `/ats/score` endpoint evaluates candidate text for Applicant Tracking Systems based on financial and analytical keywords.

### Example 1
**Input:** *(Clear Positive)*
```json
{"text": "Directed a cross-functional team to reduce operational costs by 15%."}
```
**Output:**
```json
{
  "score": 85,
  "match_level": "high",
  "keywords_found": ["reduce", "costs"]
}
```

### Example 2
**Input:** *(Neutral)*
```json
{"text": "Assisted with weekly reporting and internal audits."}
```
**Output:**
```json
{
  "score": 45,
  "match_level": "medium",
  "keywords_found": ["reporting", "audits"]
}
```

### Example 3
**Input:** *(Financial Jargon)*
```json
{"text": "Managed a $5M portfolio and optimized the EBITDA margins through aggressive restructuring."}
```
**Output:**
```json
{
  "score": 92,
  "match_level": "high",
  "keywords_found": ["portfolio", "EBITDA", "margins", "restructuring"]
}
```

## Known Limitations

- Model may occasionally misclassify highly ambiguous or sarcasm-heavy sentences.
- Performance on mixed-language (e.g. Turkish + English) heavily depends on the translation layer.
- Some niche financial jargon might not be fully captured if it was not present in the FinBERT pre-training corpus.
