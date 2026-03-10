import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Genel ayarlar ──────────────────────────────────────────────
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11
COLORS = {"negative": "#e74c3c", "neutral": "#95a5a6", "positive": "#2ecc71"}

df = pd.read_csv("data/financial_phrasebank.csv")

# ══════════════════════════════════════════════════════════════
# 1. TEMEL İSTATİSTİKLER
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("  DATASET GENEL BAKIŞ")
print("=" * 55)
print(f"Toplam örnek      : {len(df)}")
print(f"Sütunlar          : {list(df.columns)}")
print(f"Eksik değer       : {df.isnull().sum().sum()}")
print()

counts = df["label_str"].value_counts()
print("Label dağılımı:")
for label, count in counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<10} {count:>5}  ({pct:5.1f}%)  {bar}")

# ══════════════════════════════════════════════════════════════
# 2. METİN UZUNLUĞU ANALİZİ
# ══════════════════════════════════════════════════════════════
df["char_count"]  = df["sentence"].str.len()
df["word_count"]  = df["sentence"].str.split().str.len()
df["token_approx"] = (df["char_count"] / 4).astype(int)  # kaba token tahmini

print()
print("=" * 55)
print("  METİN UZUNLUĞU (kelime sayısı)")
print("=" * 55)
stats = df.groupby("label_str")["word_count"].describe()[["mean","min","50%","max"]]
print(stats.round(1).to_string())

print()
print(f"512 token'ı aşan cümle (BERT limiti): "
      f"{(df['token_approx'] > 512).sum()} adet")

# ══════════════════════════════════════════════════════════════
# 3. GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Financial Phrasebank — EDA", fontweight="bold", fontsize=13)

# — 3a. Label dağılımı (bar chart) —
ax = axes[0]
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[l] for l in counts.index], edgecolor="white", linewidth=1.5)
ax.set_title("Label Dağılımı")
ax.set_ylabel("Örnek Sayısı")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(val), ha="center", fontweight="bold")
ax.set_ylim(0, counts.max() * 1.15)
ax.spines[["top","right"]].set_visible(False)

# — 3b. Kelime sayısı dağılımı (histogram, label'a göre renkli) —
ax = axes[1]
for label in ["negative", "neutral", "positive"]:
    subset = df[df["label_str"] == label]["word_count"]
    ax.hist(subset, bins=30, alpha=0.6, color=COLORS[label], label=label, edgecolor="none")
ax.set_title("Kelime Sayısı Dağılımı")
ax.set_xlabel("Kelime Sayısı")
ax.set_ylabel("Frekans")
ax.legend()
ax.spines[["top","right"]].set_visible(False)

# — 3c. Boxplot — label başına uzunluk —
ax = axes[2]
data_to_plot = [df[df["label_str"]==l]["word_count"].values
                for l in ["negative","neutral","positive"]]
bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False,
                medianprops=dict(color="white", linewidth=2))
for patch, label in zip(bp["boxes"], ["negative","neutral","positive"]):
    patch.set_facecolor(COLORS[label])
ax.set_xticklabels(["negative","neutral","positive"])
ax.set_title("Kelime Sayısı — Boxplot")
ax.set_ylabel("Kelime Sayısı")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("data/eda_plots.png", bbox_inches="tight")
print()
print("Grafik kaydedildi: data/eda_plots.png")

# ══════════════════════════════════════════════════════════════
# 4. ÖRNEK CÜMLELER — her sınıftan 2'şer tane
# ══════════════════════════════════════════════════════════════
print()
print("=" * 55)
print("  ÖRNEK CÜMLELER")
print("=" * 55)
for label in ["negative", "neutral", "positive"]:
    print(f"\n[ {label.upper()} ]")
    samples = df[df["label_str"] == label]["sentence"].sample(2, random_state=42)
    for i, s in enumerate(samples, 1):
        print(f"  {i}. {s[:120]}{'...' if len(s)>120 else ''}")

# ══════════════════════════════════════════════════════════════
# 5. CLASS IMBALANCE — ne yapmalıyız?
# ══════════════════════════════════════════════════════════════
majority = counts.max()
print()
print("=" * 55)
print("  CLASS IMBALANCE ANALİZİ")
print("=" * 55)
for label, count in counts.items():
    ratio = majority / count
    print(f"  {label:<10}  imbalance ratio: {ratio:.2f}x")

print("""
Strateji: Fine-tuning sırasında class_weight='balanced' 
veya WeightedRandomSampler kullanacağız. (Adım 3'te ele alacağız)
""")
