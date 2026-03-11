# FinBERT Sentinel API Evaluation Report

**Date Generated:** 2026-03-11 17:52:52

## Overall Accuracy: 66.67%

## Metrics by Class

| Sentiment   |   Precision |   Recall |   F1 Score |   Support |
|:------------|------------:|---------:|-----------:|----------:|
| Positive    |        0.73 |     0.89 |       0.8  |         9 |
| Negative    |        0.62 |     0.56 |       0.59 |         9 |
| Neutral     |        0.64 |     0.58 |       0.61 |        12 |

## Category Performance

| Category         | Accuracy   |
|:-----------------|:-----------|
| ambiguous        | 20.00%     |
| clear_negative   | 60.00%     |
| clear_neutral    | 100.00%    |
| clear_positive   | 100.00%    |
| financial_jargon | 20.00%     |
| mixed_language   | 100.00%    |

## Misclassifications

| id     | category         | expected   | predicted   |   confidence |
|:-------|:-----------------|:-----------|:------------|-------------:|
| neg_03 | clear_negative   | negative   | neutral     |       0.9505 |
| neg_04 | clear_negative   | negative   | neutral     |       0.9945 |
| amb_01 | ambiguous        | neutral    | positive    |       0.9909 |
| amb_02 | ambiguous        | neutral    | negative    |       0.9812 |
| amb_03 | ambiguous        | neutral    | negative    |       0.9289 |
| amb_04 | ambiguous        | neutral    | negative    |       0.9858 |
| jar_02 | financial_jargon | negative   | neutral     |       0.9573 |
| jar_03 | financial_jargon | positive   | neutral     |       0.9905 |
| jar_04 | financial_jargon | negative   | positive    |       0.992  |
| jar_05 | financial_jargon | neutral    | positive    |       0.9863 |
