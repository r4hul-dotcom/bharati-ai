from pymongo import MongoClient
from pathlib import Path
import joblib
import string
from datetime import datetime
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz
import nltk

from . import db, collection, leads_collection, campaigns_collection, sent_emails_collection, replies_collection, users_collection, email_replies_collection, paired_emails_collection, legacy_leads_collection, legacy_campaigns_collection

nltk.download('punkt')

# === Load Classifier ===
model_path = Path(__file__).parent / "spam_detection_model.joblib"
try:
    classifier_data = joblib.load(model_path)
    model = classifier_data["model"]
    vectorizer = classifier_data["vectorizer"]
    if not hasattr(model, "predict_proba"):
        raise Exception("Model does not support predict_proba")
except Exception as e:
    print(f"ML model load error: {e}")
    model, vectorizer = None, None

# === Rule Keywords ===
categories = {
    "quotation_request": ["quotation", "quote", "price", "estimate", "cost", "rate"],
    "complaint": ["complaint", "problem", "issue", "damaged", "not working", "faulty", "disappointed", "bad product", "does not work", "malfunction", "refund", "return"],
    "follow_up": ["follow up", "reminder", "status", "pending", "update", "any update"],
    "feedback": ["feedback", "review", "experience", "suggestion", "thank you", "appreciate", "great job", "awesome", "satisfied", "unsatisfied", "excellent", "amazing", "good service", "support was great", "really helpful", "happy with"],
    "sales_approval": ["approval", "approve", "authorize", "purchase order"],
    "dispatch_update": ["dispatch", "delivery", "courier", "shipment", "track", "lr copy"],
    "general_enquiry": ["enquiry", "inquiry", "information", "clarification", "need details"]
}

# === Helpers ===
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def smart_pattern_fallback(text):
    text = text.lower()
    patterns = {
        "quotation_request": ["please send quotation", "need a quote", "request for quotation", "provide the rates", "price list", "send cost", "need price", "quote us", "share pricing", "quotation for", "rfq", "rate contract", "commercial offer"],
        "complaint": ["not working", "damaged", "leaking", "defective", "faulty", "not satisfied", "wrong item", "complaint", "need replacement", "issue with", "return this", "broken", "malfunction"],
        "follow_up": ["just following up", "any update", "still waiting", "haven’t heard", "reminder", "status of our previous mail", "awaiting your reply", "follow up", "please revert"],
        "dispatch_update": ["dispatched", "tracking id", "awb number", "courier", "lr copy", "shipment", "delivery update", "logistics", "delivery status", "dispatch details"],
        "feedback": ["thank you", "appreciate", "good job", "great service", "helpful", "satisfied", "excellent service", "feedback", "recommend", "testimonials", "very satisfied", "happy with"],
        "sales_approval": ["approve the po", "approval required", "awaiting approval", "need your approval", "approval from finance", "please approve", "approval needed", "po confirmation"],
        "general_enquiry": ["enquiry", "want information", "details about", "can you share", "product specification", "availability of", "lead time", "need info", "clarification"]
    }
    for category, phrases in patterns.items():
        for phrase in phrases:
            if fuzz.partial_ratio(text, phrase) > 85:
                return category
    return None


def sentence_level_fallback(text):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split(".")
    votes = {}
    for sentence in sentences:
        cat = smart_pattern_fallback(sentence.strip())
        if cat:
            votes[cat] = votes.get(cat, 0) + 1
    return max(votes.items(), key=lambda x: x[1])[0] if votes else "general_enquiry"


def classify_email(email_text):
    cleaned = preprocess(email_text)
    ml_prediction = None
    ml_confidence = 0.0
    if model and vectorizer:
        try:
            X = vectorizer.transform([cleaned])
            proba = model.predict_proba(X)[0]
            max_idx = proba.argmax()
            ml_prediction = model.classes_[max_idx]
            ml_confidence = proba[max_idx]
        except Exception as e:
            print(f"ML prediction failed: {e}")
    rule_prediction = smart_pattern_fallback(cleaned)
    rule_confidence = 0.0
    if rule_prediction:
        rule_confidence = sum([
            fuzz.partial_ratio(cleaned, phrase)
            for phrase in categories.get(rule_prediction, [])
            if fuzz.partial_ratio(cleaned, phrase) > 80
        ]) / 1000

    if ml_prediction == rule_prediction and ml_prediction:
        final_prediction = ml_prediction
    elif rule_confidence >= 0.5:
        final_prediction = rule_prediction
    elif ml_confidence >= 0.6:
        final_prediction = ml_prediction
    else:
        final_prediction = sentence_level_fallback(email_text)

    return final_prediction, {
        "ml_prediction": ml_prediction,
        "ml_confidence": round(ml_confidence, 3),
        "rule_prediction": rule_prediction,
        "rule_confidence": round(rule_confidence, 3)
    }


# === Main Logic ===
def label_paired_emails():

    count = 0
    for doc in collection.find():
        original_text = doc.get("email_text", "")
        if not original_text.strip():
            continue

        predicted_category, metadata = classify_email(original_text)

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "predicted_category": predicted_category,
                "ml_prediction": metadata["ml_prediction"],
                "ml_confidence": metadata["ml_confidence"],
                "rule_prediction": metadata["rule_prediction"],
                "rule_confidence": metadata["rule_confidence"],
                "classified_at": datetime.now()
            }}
        )
        count += 1
        print(f"[✓] Labeled: {predicted_category} → {original_text[:60]}...")

    print(f"\n✅ Total Labeled Emails: {count}")

if __name__ == "__main__":
    label_paired_emails()
