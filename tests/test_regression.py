import pytest
import httpx
import json
import os
import pandas as pd
from datetime import datetime

# URL of the local API for evaluation
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
REPORT_MODE = os.getenv("REPORT_MODE", "true").lower() == "true"
REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "eval_report.md")

# Load evaluation data
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Optional: Add a hook to collect results across all tests
results_log = []

@pytest.mark.asyncio
@pytest.mark.parametrize("item", eval_data, ids=[item["id"] for item in eval_data])
async def test_sentiment_prediction(item):
    """
    Test individual sentiment predictions loaded from the evaluation dataset.
    We assert that the API response matches the expected manual label.
    """
    text = item["text"]
    expected = item["expected_sentiment"]
    category = item["category"]
    
    # 1. Send Request
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/predict", json={"text": text})
    
    # 2. Check basic HTTP status
    assert response.status_code == 200, f"API failed with {response.status_code}: {response.text}"
    
    data = response.json()
    predicted = data.get("sentiment", "unknown")
    confidence = data.get("confidence", 0.0)
    
    # 3. Save result for report generation
    results_log.append({
        "id": item["id"],
        "text_preview": text[:40] + "..." if len(text) > 40 else text,
        "category": category,
        "expected": expected,
        "predicted": predicted,
        "confidence": confidence,
        "correct": (expected == predicted)
    })
    
    # 4. Check label correctness
    assert expected == predicted, f"[{category}] Expected {expected}, got {predicted}. Text: {text}"
    
    # 5. Check confidence thresholds
    # We expect clear cases to have higher confidence than ambiguous ones
    if "clear" in category:
        assert confidence > 0.50, f"Confidence too low ({confidence:.2f}) for clear category: {category}"


@pytest.fixture(scope="session", autouse=True)
def teardown_evaluation_report():
    """
    Session teardown fixture that generates the markdown report
    if REPORT_MODE is enabled.
    """
    yield # Let all tests run first
    
    if REPORT_MODE and results_log:
        df = pd.DataFrame(results_log)
        
        # Calculate metrics per class
        classes = ["positive", "negative", "neutral"]
        class_metrics = []
        
        for cls in classes:
            # True Positives: Predicted cls AND Actual cls
            tp = len(df[(df["predicted"] == cls) & (df["expected"] == cls)])
            # False Positives: Predicted cls AND Actual NOT cls
            fp = len(df[(df["predicted"] == cls) & (df["expected"] != cls)])
            # False Negatives: Predicted NOT cls AND Actual cls
            fn = len(df[(df["predicted"] != cls) & (df["expected"] == cls)])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics.append({
                "Sentiment": cls.capitalize(),
                "Precision": f"{precision:.2f}",
                "Recall": f"{recall:.2f}",
                "F1 Score": f"{f1:.2f}",
                "Support": len(df[df["expected"] == cls])
            })
            
        metrics_df = pd.DataFrame(class_metrics)
        overall_accuracy = len(df[df["correct"] == True]) / len(df)
        
        # Write to Markdown
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("# FinBERT Sentinel API Evaluation Report\n\n")
            f.write(f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Overall Accuracy: {overall_accuracy:.2%}\n\n")
            f.write("## Metrics by Class\n\n")
            f.write(metrics_df.to_markdown(index=False))
            f.write("\n\n")
            
            f.write("## Category Performance\n\n")
            cat_perf = df.groupby('category')['correct'].mean().reset_index()
            cat_perf['correct'] = cat_perf['correct'].apply(lambda x: f"{x:.2%}")
            cat_perf.rename(columns={'category': 'Category', 'correct': 'Accuracy'}, inplace=True)
            f.write(cat_perf.to_markdown(index=False))
            f.write("\n\n")
            
            # Optional: Error analysis section (show false predictions)
            errors = df[df["correct"] == False]
            if not errors.empty:
                f.write("## Misclassifications\n\n")
                f.write(errors[['id', 'category', 'expected', 'predicted', 'confidence']].to_markdown(index=False))
                f.write("\n")
                
        print(f"\n[Info] Evaluation report generated at: {REPORT_PATH}")
