import joblib
import numpy as np
import pandas as pd
import re
from pathlib import Path
from scipy.sparse import hstack
import os

# Load model components
model = joblib.load("xgboost_email_classifier.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Structured feature extraction
def extract_structured_features(text):
    return np.array([[
        len(text),
        text.count('?'),
        int('description' in text and 'qty' in text or ':' in text),
        int(bool(re.search(r'(PO[\s#:/\-]*\d+)', text, re.IGNORECASE))),
        int(bool(re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b', text))),
        int(bool(re.search(r'model[\s:]*[A-Z0-9\-]+', text, re.IGNORECASE)))
    ]])

# Read input from .csv or .txt
def read_input_file(filepath):
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        col = "email_text" if "email_text" in df.columns else "text"
        return df[col].dropna().tolist()
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        raise ValueError("Only .csv or .txt files are supported")


# Predict in batch
def predict_batch(emails):
    results = []
    for email in emails:
        cleaned = email.lower().strip()
        X_text = vectorizer.transform([cleaned])
        structured = extract_structured_features(cleaned)
        X_final = hstack([X_text, structured])
        proba = model.predict_proba(X_final)[0]
        pred = label_encoder.inverse_transform([np.argmax(proba)])[0]
        confidence = round(np.max(proba), 3)
        results.append((email, pred, confidence))
    return results

# Run the pipeline
if __name__ == "__main__":
    input_path = input("📂 Enter path to your .csv or .txt file: ").strip()

    if not os.path.isfile(input_path):
        print("❌ File not found. Please check the path and try again.")
    else:
        try:
            emails = read_input_file(input_path)
            predictions = predict_batch(emails)
            print("\n📦 Batch Predictions:\n")
            for i, (email, category, conf) in enumerate(predictions, 1):
                print(f"{i}. {email}")
                print(f"   → Predicted: {category} (confidence: {conf})\n")
        except Exception as e:
            print(f"❌ Error: {e}")
