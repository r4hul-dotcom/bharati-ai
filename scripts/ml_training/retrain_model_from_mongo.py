import pandas as pd
import string
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re

from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

# --- Load Data ---
data = list(collection.find({"category": {"$exists": True}}))
df = pd.DataFrame(data)
df = df[["email_text", "category"]].dropna()
df = df[df["category"] != "other"]  # remove 'other' if used as fallback

# --- Preprocess ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df["clean_text"] = df["email_text"].apply(clean_text)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["category"], stratify=df["category"], test_size=0.2, random_state=42)

# --- Vectorize ---
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Train Model ---
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_vec)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# --- Save Model ---
joblib.dump({"model": model, "vectorizer": vectorizer}, "spam_detection_model.joblib")
print("\n✅ Model saved as spam_detection_model.joblib")
