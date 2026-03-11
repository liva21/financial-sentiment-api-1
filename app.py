import gradio as gr
import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
)
import numpy as np

# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────
FINBERT_PATH = "./models/finbert-finetuned"
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-tr-en"

# ─────────────────────────────────────────────
# LOAD MODELS (cached after first run)
# ─────────────────────────────────────────────
print("Loading FinBERT model...")
try:
    finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_PATH)
    finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_PATH)
    finbert_model.eval()
    FINBERT_LABELS = list(finbert_model.config.id2label.values())
except Exception as e:
    print(f"[WARN] Could not load local FinBERT, falling back to ProsusAI/finbert: {e}")
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert_model.eval()
    FINBERT_LABELS = ["positive", "negative", "neutral"]

print("Loading translation model...")
tr_tokenizer = MarianTokenizer.from_pretrained(TRANSLATE_MODEL)
tr_model = MarianMTModel.from_pretrained(TRANSLATE_MODEL)
tr_model.eval()
print("All models loaded.")

# ─────────────────────────────────────────────
# FINANCIAL KEYWORDS (EN)
# ─────────────────────────────────────────────
FINANCIAL_KEYWORDS = [
    "revenue", "profit", "loss", "earnings", "growth", "decline", "risk",
    "investment", "market", "stock", "bond", "interest", "rate", "inflation",
    "debt", "equity", "dividend", "volatility", "forecast", "outlook",
    "recession", "expansion", "gdp", "cash", "flow", "asset", "liability",
    "bankruptcy", "merger", "acquisition", "ipo", "shares", "fund",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def detect_language(text: str) -> str:
    """Simple heuristic: Turkish-specific characters → 'tr', else 'en'."""
    tr_chars = set("çğıöşüÇĞİÖŞÜ")
    if any(c in tr_chars for c in text):
        return "tr"
    turkish_words = {"ve", "bir", "bu", "ile", "için", "da", "de", "den", "nin",
                     "nın", "nun", "nün", "ın", "in", "un", "ün", "yı", "yi",
                     "yu", "yü", "ta", "te", "tan", "ten"}
    words = set(text.lower().split())
    if len(words & turkish_words) >= 2:
        return "tr"
    return "en"


def translate_tr_to_en(text: str) -> str:
    inputs = tr_tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = tr_model.generate(**inputs)
    return tr_tokenizer.decode(translated[0], skip_special_tokens=True)


def extract_keywords(text: str) -> list[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    found = [w for w in words if w in FINANCIAL_KEYWORDS]
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def get_risk_level(label: str, confidence: float) -> str:
    label = label.lower()
    if label == "negative":
        if confidence >= 0.80:
            return "🔴 HIGH RISK"
        elif confidence >= 0.55:
            return "🟠 MEDIUM RISK"
        else:
            return "🟡 LOW-MEDIUM RISK"
    elif label == "positive":
        if confidence >= 0.80:
            return "🟢 LOW RISK"
        else:
            return "🟡 LOW-MEDIUM RISK"
    else:
        return "🟡 NEUTRAL / MONITOR"


def run_finbert(text: str):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512, padding=True)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    idx = int(np.argmax(probs))
    label = FINBERT_LABELS[idx]
    confidence = float(probs[idx])
    return label, confidence, probs


# ─────────────────────────────────────────────
# MAIN PREDICT FUNCTION
# ─────────────────────────────────────────────

def analyze(text: str):
    if not text or not text.strip():
        return "⚠️ Please enter some text.", "", "", "", ""

    lang = detect_language(text)
    original_text = text

    if lang == "tr":
        translated_text = translate_tr_to_en(text)
        lang_info = f"🌐 Detected: **Turkish** → translated to English"
    else:
        translated_text = text
        lang_info = "🌐 Detected: **English**"

    label, confidence, all_probs = run_finbert(translated_text)
    risk = get_risk_level(label, confidence)
    keywords = extract_keywords(translated_text)

    sentiment_emoji = {"positive": "📈", "negative": "📉", "neutral": "➡️"}
    emoji = sentiment_emoji.get(label.lower(), "❓")

    label_display = f"{emoji} {label.upper()}"
    confidence_display = f"{confidence*100:.1f}%"
    keywords_display = ", ".join(keywords) if keywords else "—"

    # Build score breakdown
    scores_md = "\n".join(
        [f"- **{FINBERT_LABELS[i]}**: {all_probs[i]*100:.1f}%"
         for i in range(len(FINBERT_LABELS))]
    )

    translation_note = (
        f"\n\n**Translated text:** _{translated_text}_"
        if lang == "tr" else ""
    )

    summary = (
        f"{lang_info}{translation_note}\n\n"
        f"### Score Breakdown\n{scores_md}"
    )

    return label_display, confidence_display, risk, keywords_display, summary


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="Financial Sentiment Analysis API",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .result-box { border-radius: 8px; padding: 8px; }
    footer { display: none !important; }
    """,
) as demo:

    gr.Markdown(
        """
        # 📊 Financial Sentiment Analysis
        ### Powered by FinBERT · Supports Turkish & English
        Paste any financial news headline, earnings summary, or analyst comment.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📝 Input Text (Turkish or English)",
                placeholder="e.g. 'Company reported record profits this quarter' or 'Şirket bu çeyrekte rekor kar açıkladı'",
                lines=5,
            )
            submit_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_label = gr.Textbox(label="Sentiment Label", elem_classes="result-box")
            out_confidence = gr.Textbox(label="Confidence Score", elem_classes="result-box")
            out_risk = gr.Textbox(label="Risk Level", elem_classes="result-box")
            out_keywords = gr.Textbox(label="Financial Keywords", elem_classes="result-box")

    out_summary = gr.Markdown(label="Details")

    submit_btn.click(
        fn=analyze,
        inputs=[text_input],
        outputs=[out_label, out_confidence, out_risk, out_keywords, out_summary],
    )

    gr.Examples(
        examples=[
            ["The company reported a significant drop in quarterly earnings due to supply chain disruptions."],
            ["Strong revenue growth and expanding margins signal a bullish outlook for investors."],
            ["Şirketin hisse senetleri, beklentilerin üzerinde kar açıklamasının ardından yükseldi."],
            ["Merkez bankası faiz oranlarını artırarak enflasyonla mücadele etmeye devam ediyor."],
            ["Markets remained flat as investors awaited the Federal Reserve's rate decision."],
        ],
        inputs=text_input,
        label="📌 Example Inputs",
    )

    gr.Markdown(
        """
        ---
        **Model:** Fine-tuned FinBERT for financial sentiment classification  
        **Translation:** Helsinki-NLP/opus-mt-tr-en for Turkish→English  
        **Labels:** Positive · Negative · Neutral
        """
    )

if __name__ == "__main__":
    demo.launch()
