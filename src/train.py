import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import set_seed

# ══════════════════════════════════════════════════════════════
# CONFIG — tek yerden yönet
# ══════════════════════════════════════════════════════════════
CONFIG = {
    "model_name"    : "ProsusAI/finbert",   # finansal domaine pre-train edilmiş BERT
    "max_length"    : 128,                  # cümlelerimiz ~22 kelime, 128 yeterli
    "batch_size"    : 16,
    "epochs"        : 5,
    "lr"            : 2e-5,                 # BERT fine-tune için standart aralık
    "seed"          : 42,
    "subset_size"   : None,                 # None = tüm veri, int = ilk N örnek
    "output_dir"    : "models/finbert-finetuned",
    "label2id"      : {"negative": 0, "neutral": 1, "positive": 2},
    "id2label"      : {0: "negative", 1: "neutral", 2: "positive"},
}

set_seed(CONFIG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Kullanılan cihaz: {DEVICE}")

# ══════════════════════════════════════════════════════════════
# 1. VERİ YÜKLE & BÖLE
# ══════════════════════════════════════════════════════════════
df = pd.read_csv("data/financial_phrasebank.csv")
df["label"] = df["label_str"].map(CONFIG["label2id"])

if CONFIG["subset_size"]:
    df = df.sample(CONFIG["subset_size"], random_state=CONFIG["seed"])

# Stratified split — her sınıf oranı korunur
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=CONFIG["seed"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=CONFIG["seed"]
)

print(f"\nVeri boyutları:")
print(f"  Train : {len(train_df)}")
print(f"  Val   : {len(val_df)}")
print(f"  Test  : {len(test_df)}")

# ══════════════════════════════════════════════════════════════
# 2. TOKENİZASYON
# ══════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=CONFIG["max_length"],
    )

def df_to_dataset(df):
    ds = Dataset.from_pandas(df[["sentence", "label"]].reset_index(drop=True))
    return ds.map(tokenize, batched=True)

print("\nTokenizer yükleniyor ve veri tokenize ediliyor...")
train_ds = df_to_dataset(train_df)
val_ds   = df_to_dataset(val_df)
test_ds  = df_to_dataset(test_df)

# Test setini ileride kullanmak üzere kaydet
test_df.to_csv("data/test_set.csv", index=False)
print("Test seti kaydedildi: data/test_set.csv")

# ══════════════════════════════════════════════════════════════
# 3. CLASS WEIGHT — imbalance'a karşı
# ══════════════════════════════════════════════════════════════
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_df["label"].values,
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"\nClass weights: {dict(zip(['neg','neu','pos'], class_weights.round(2)))}")

# ══════════════════════════════════════════════════════════════
# 4. MODEL
# ══════════════════════════════════════════════════════════════
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=3,
    id2label=CONFIG["id2label"],
    label2id=CONFIG["label2id"],
    ignore_mismatched_sizes=True,   # FinBERT'in orijinal head'i 3 label, yine de safe
)
model = model.to(DEVICE)

# ══════════════════════════════════════════════════════════════
# 5. CUSTOM TRAINER — weighted loss
# ══════════════════════════════════════════════════════════════
class WeightedTrainer(Trainer):
    """Class imbalance'ı kompanse etmek için weighted CrossEntropy."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ══════════════════════════════════════════════════════════════
# 6. METRİKLER
# ══════════════════════════════════════════════════════════════
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

# ══════════════════════════════════════════════════════════════
# 7. TRAINING ARGUMENTS
# ══════════════════════════════════════════════════════════════
args = TrainingArguments(
    output_dir                  = CONFIG["output_dir"],
    num_train_epochs            = CONFIG["epochs"],
    per_device_train_batch_size = CONFIG["batch_size"],
    per_device_eval_batch_size  = CONFIG["batch_size"],
    learning_rate               = CONFIG["lr"],
    weight_decay                = 0.01,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1_macro",
    greater_is_better           = True,
    logging_steps               = 20,
    seed                        = CONFIG["seed"],
    report_to                   = "none",       # W&B vs. kapalı
    fp16                        = DEVICE == "cuda",  # GPU varsa hızlandır
)

# ══════════════════════════════════════════════════════════════
# 8. TRAIN!
# ══════════════════════════════════════════════════════════════
trainer = WeightedTrainer(
    model           = model,
    args            = args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n" + "="*55)
print("  EĞİTİM BAŞLIYOR")
print("="*55)
trainer.train()

# ══════════════════════════════════════════════════════════════
# 9. KAYDET
# ══════════════════════════════════════════════════════════════
os.makedirs(CONFIG["output_dir"], exist_ok=True)
trainer.save_model(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"\nModel kaydedildi: {CONFIG['output_dir']}")

# Son val metriklerini göster
final = trainer.evaluate()
print("\nFinal Validation Metrikleri:")
for k, v in final.items():
    if isinstance(v, float):
        print(f"  {k:<25} {v:.4f}")
