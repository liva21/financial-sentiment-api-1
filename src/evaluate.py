import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
MODEL_DIR  = "models/finbert-finetuned"
TEST_CSV   = "data/test_set.csv"
ID2LABEL   = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID   = {"negative": 0, "neutral": 1, "positive": 2}
MAX_LENGTH = 128
BATCH_SIZE = 32
COLORS     = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#2ecc71"}

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Cihaz: {DEVICE}")

# ══════════════════════════════════════════════════════════════
# 1. MODEL & TOKENİZER YÜKLE
# ══════════════════════════════════════════════════════════════
print("Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# ══════════════════════════════════════════════════════════════
# 2. TAHMİN FONKSİYONU
# ══════════════════════════════════════════════════════════════
def predict(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Batch tahmin — logits ve predicted label id'leri döner."""
    all_preds, all_probs = [], []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc   = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_preds.append(preds)
        all_probs.append(probs)

    return np.concatenate(all_preds), np.vstack(all_probs)

# ══════════════════════════════════════════════════════════════
# 3. TEST SETİ TAHMİNLERİ
# ══════════════════════════════════════════════════════════════
df        = pd.read_csv(TEST_CSV)
df["label"] = df["label_str"].map(LABEL2ID)

print(f"Test seti: {len(df)} örnek")
preds, probs = predict(df["sentence"].tolist())

df["pred"]       = preds
df["pred_str"]   = df["pred"].map(ID2LABEL)
df["confidence"] = probs.max(axis=1)
df["correct"]    = df["label"] == df["pred"]

# ══════════════════════════════════════════════════════════════
# 4. METRİKLER
# ══════════════════════════════════════════════════════════════
acc        = accuracy_score(df["label"], df["pred"])
f1_macro   = f1_score(df["label"], df["pred"], average="macro")
f1_weighted= f1_score(df["label"], df["pred"], average="weighted")

print("\n" + "="*55)
print("  TEST METRİKLERİ")
print("="*55)
print(f"  Accuracy      : {acc:.4f}")
print(f"  F1 Macro      : {f1_macro:.4f}")
print(f"  F1 Weighted   : {f1_weighted:.4f}")

print("\n--- Classification Report ---")
print(classification_report(
    df["label"], df["pred"],
    target_names=["negative", "neutral", "positive"]
))

# ══════════════════════════════════════════════════════════════
# 5. GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Değerlendirme — Test Seti", fontweight="bold")

# — 5a. Confusion Matrix —
cm     = confusion_matrix(df["label"], df["pred"])
labels = ["negative", "neutral", "positive"]
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=labels, yticklabels=labels,
    ax=axes[0], cbar=False,
    annot_kws={"size": 13, "weight": "bold"},
)
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Gerçek")
axes[0].set_xlabel("Tahmin")

# — 5b. Confidence dağılımı (doğru vs. yanlış) —
ax = axes[1]
for correct, label, color in [(True, "Doğru", "#2ecc71"), (False, "Yanlış", "#e74c3c")]:
    subset = df[df["correct"] == correct]["confidence"]
    ax.hist(subset, bins=20, alpha=0.7, color=color, label=f"{label} ({len(subset)})")
ax.set_title("Tahmin Güven Skoru")
ax.set_xlabel("Confidence (softmax max)")
ax.set_ylabel("Frekans")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# — 5c. Sınıf bazında F1 —
ax    = axes[2]
report = classification_report(
    df["label"], df["pred"],
    target_names=labels, output_dict=True
)
f1s   = [report[l]["f1-score"] for l in labels]
bars  = ax.bar(labels, f1s, color=[COLORS[l] for l in labels], edgecolor="white")
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontweight="bold")
ax.set_title("Sınıf Bazında F1 Skoru")
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Score")
ax.axhline(y=f1_macro, color="gray", linestyle="--", alpha=0.7, label=f"Macro avg: {f1_macro:.3f}")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("data/evaluation_plots.png", bbox_inches="tight")
print("Grafik kaydedildi: data/evaluation_plots.png")

# ══════════════════════════════════════════════════════════════
# 6. HATA ANALİZİ — modelin yanıldığı örnekler
# ══════════════════════════════════════════════════════════════
errors = df[~df["correct"]].sort_values("confidence", ascending=False)

print(f"\n{'='*55}")
print(f"  HATA ANALİZİ — {len(errors)} yanlış tahmin")
print(f"{'='*55}")

if len(errors) > 0:
    print(f"\nEn güvenli yanlış tahminler (yüksek confidence ama yanlış):")
    for _, row in errors.head(5).iterrows():
        print(f"\n  Cümle     : {row['sentence'][:100]}...")
        print(f"  Gerçek    : {row['label_str']:<10}  Tahmin: {row['pred_str']:<10}  Conf: {row['confidence']:.3f}")
else:
    print("Hata yok — mükemmel test performansı!")

# Hataları kaydet
errors[["sentence","label_str","pred_str","confidence"]].to_csv(
    "data/errors.csv", index=False
)
print(f"\nHatalar kaydedildi: data/errors.csv")
