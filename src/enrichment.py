"""
Sentiment sonucunu zenginleştiren modül.
- Keyword extraction: kararı etkileyen anahtar kelimeler
- Risk skoru: finansal risk seviyesi (0-1)
"""
import re

# ── Finansal anahtar kelime sözlükleri ────────────────────────
NEGATIVE_KEYWORDS = [
    # Zarar / Kayıp
    "loss", "losses", "deficit", "decline", "fell", "fallen", "drop", "dropped",
    "decrease", "decreased", "plunge", "plunged", "slump", "slumped",
    "bankruptcy", "bankrupt", "insolvent", "default", "defaulted",
    # İşten çıkarma
    "layoff", "layoffs", "redundan", "restructur", "downsize",
    # Düşüş
    "downgrade", "downgraded", "cut", "cuts", "weak", "weakened",
    "disappoint", "miss", "missed", "below", "shortfall",
    # Risk
    "risk", "warning", "concern", "uncertain", "volatil",
    "crash", "crashing", "crisis", "recession", "inflation",
]

POSITIVE_KEYWORDS = [
    # Kazanç / Büyüme
    "profit", "profits", "gain", "gains", "growth", "grew", "surge", "surged",
    "rise", "rose", "risen", "increase", "increased", "jump", "jumped",
    "soar", "soared", "climb", "climbed", "rally", "rallied",
    # Başarı
    "record", "beat", "beats", "exceeded", "outperform", "strong", "stronger",
    "robust", "solid", "improve", "improved", "expand", "expanded",
    # Pozitif gelişmeler
    "acquire", "acquisition", "partnership", "deal", "contract", "award",
    "dividend", "buyback", "upgrade", "upgraded",
]

NEUTRAL_KEYWORDS = [
    "maintain", "maintained", "unchanged", "stable", "steady",
    "flat", "hold", "held", "in line", "meet", "met", "expect",
    "announce", "announced", "report", "reported", "said", "stated",
]

# ── Risk faktörleri (ağırlıklı) ───────────────────────────────
HIGH_RISK_TERMS = {
    "bankruptcy": 1.0, "bankrupt": 1.0, "default": 0.95,
    "crisis": 0.90, "crash": 0.90, "recession": 0.85,
    "fraud": 0.95, "investigation": 0.80, "lawsuit": 0.75,
    "layoff": 0.70, "restructur": 0.70, "downgrade": 0.65,
    "loss": 0.60, "decline": 0.55, "drop": 0.50,
}

def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """
    Metindeki finansal anahtar kelimeleri çıkar.
    Hem pozitif hem negatif kelimeleri döner, sentiment'e göre sıralar.
    """
    text_lower = text.lower()
    words      = re.findall(r'\b\w+\b', text_lower)
    found      = []

    all_keywords = NEGATIVE_KEYWORDS + POSITIVE_KEYWORDS + NEUTRAL_KEYWORDS

    for word in words:
        for kw in all_keywords:
            if kw in word and word not in found:
                found.append(word)
                break

    # Bulunamazsa en uzun kelimeleri döndür (fallback)
    if not found:
        found = sorted(set(words), key=len, reverse=True)[:top_n]

    return found[:top_n]

def calculate_risk_score(
    text: str,
    sentiment: str,
    confidence: float,
) -> tuple[float, str]:
    """
    Risk skorunu hesapla (0.0 - 1.0).

    Faktörler:
    - Sentiment: negative yüksek risk, positive düşük risk
    - Confidence: yüksek confidence → daha kesin risk
    - Yüksek riskli terimler: bankruptcy, crisis vb.

    Returns: (risk_score, risk_level)
    """
    text_lower = text.lower()

    # 1. Sentiment bazlı baz skor
    base_scores = {"negative": 0.65, "neutral": 0.35, "positive": 0.15}
    base        = base_scores[sentiment]

    # 2. Confidence ile ağırlıklandır
    weighted = base * confidence

    # 3. Yüksek riskli terim bonusu
    term_bonus = 0.0
    for term, weight in HIGH_RISK_TERMS.items():
        if term in text_lower:
            term_bonus = max(term_bonus, weight * 0.3)  # en yüksek terimi al

    # 4. Final skor (0-1 arasında kapat)
    risk_score = min(weighted + term_bonus, 1.0)
    risk_score = round(risk_score, 3)

    # 5. Risk seviyesi
    if risk_score >= 0.70:
        risk_level = "HIGH"
    elif risk_score >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_score, risk_level

def enrich(
    text: str,
    sentiment: str,
    confidence: float,
) -> dict:
    """Ana enrichment fonksiyonu — keyword + risk döner."""
    keywords             = extract_keywords(text)
    risk_score, risk_level = calculate_risk_score(text, sentiment, confidence)

    return {
        "keywords"  : keywords,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }
