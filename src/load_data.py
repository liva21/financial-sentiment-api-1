from datasets import load_dataset
import pandas as pd
import os

def load_financial_phrasebank(save_path="data/financial_phrasebank.csv"):
    """
    financial_phrasebank dataset'ini HuggingFace'den çeker.
    
    Label mapping:
        0 -> negative
        1 -> neutral  
        2 -> positive
    """
    print("Dataset yükleniyor...")
    
    # sentences_allagree: tüm annotator'ların hemfikir olduğu subset
    # En temiz veri bu — başlangıç için ideal
    dataset = load_dataset(
        "financial_phrasebank",
        "sentences_allagree",
        trust_remote_code=True
    )
    
    # HuggingFace dataset -> pandas DataFrame
    df = dataset["train"].to_pandas()
    
    # Label integer'larını okunabilir string'e çevir
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    df["label_str"] = df["label"].map(label_map)
    
    # Kaydet
    os.makedirs("data", exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Dataset kaydedildi: {save_path}")
    print(f"Toplam örnek sayısı: {len(df)}")
    print(f"\nLabel dağılımı:")
    print(df["label_str"].value_counts())
    
    return df

if __name__ == "__main__":
    df = load_financial_phrasebank()
    print("\nİlk 3 örnek:")
    print(df.head(3).to_string())
