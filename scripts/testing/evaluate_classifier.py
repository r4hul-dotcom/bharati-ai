import pandas as pd
import string
import joblib
from rapidfuzz import fuzz
from nltk.tokenize import sent_tokenize
from pathlib import Path

df = pd.read_csv("labeled_email_samples.csv")

categories = {
    "quotation_request": ["quotation", "quote", "price", "estimate", "cost", "rate"],
    "complaint": ["complaint", "problem", "issue", "damaged", "not working", "faulty", "disappointed", "bad product", "does not work", "malfunction", "refund", "return"],
    "follow_up": ["follow up", "reminder", "status", "pending", "update", "any update"],
    "feedback": ["feedback", "review", "experience", "suggestion", "thank you", "appreciate", "great job", "awesome", "satisfied", "unsatisfied", "excellent", "amazing", "good service", "support was great", "really helpful", "happy with"],
    "sales_approval": ["approval", "approve", "authorize", "purchase order"],
    "dispatch_update": ["dispatch", "delivery", "courier", "shipment", "track", "lr copy"],
    "general_enquiry": ["enquiry", "inquiry", "information", "clarification", "need details"]
}

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def smart_pattern_fallback(text):
    text = text.lower()
    if any(p in text for p in ["please send quotation", "need a quote", "request for quotation", "provide the rates", "share pricing", "quotation for", "rfq", "rate contract", "commercial offer", "price list", "send cost", "need price", "send quotation", "costing for", "share cost sheet", "price quotation", "quote us"]):
        return "quotation_request"
    if any(p in text for p in ["not working", "damaged", "leaking", "defective", "faulty", "not satisfied", "wrong item", "complaint", "need replacement", "issue with", "return this", "received broken", "product damaged", "fire extinguisher not working", "defective item", "damaged during delivery"]):
        return "complaint"
    if any(p in text for p in ["just following up", "any update", "still waiting", "haven’t heard", "reminder", "status of our previous mail", "checking back", "awaiting your reply", "when will it be done", "second reminder", "please revert", "follow up"]):
        return "follow_up"
    if any(p in text for p in ["dispatched", "tracking id", "awb number", "courier", "lr copy", "shipment", "delivery update", "has the order shipped", "kindly share tracking", "delivery status", "dispatch details", "awaiting lr copy"]):
        return "dispatch_update"
    if sum(1 for phrase in ["thank you", "appreciate", "good job", "great service", "helpful", "satisfied", "excellent service", "feedback", "recommend", "testimonials", "amazing support", "impressive service", "we are happy", "very satisfied", "you did a great job"] if phrase in text) >= 2:
        return "feedback"
    if any(p in text for p in ["approve the po", "approval required", "awaiting approval", "need your approval", "approval from finance", "please approve", "requesting approval", "approval for quotation", "approval needed", "sales manager approval", "po confirmation", "purchase approval", "waiting for approval", "kindly authorize"]):
        return "sales_approval"
    if any(p in text for p in ["enquiry", "want information", "details about", "can you share", "product specification", "availability of", "lead time", "need info", "basic enquiry", "clarification needed", "have a query", "can you tell me", "need some info", "general question"]):
        return "general_enquiry"
    return "general_enquiry"

model_path = Path("spam_detection_model.joblib")
classifier_data = joblib.load(model_path)
model = classifier_data["model"]
vectorizer = classifier_data["vectorizer"]

def classify_email(text):
    cleaned = preprocess(text)
    try:
        X = vectorizer.transform([cleaned])
        probs = model.predict_proba(X)[0]
        ml_pred = model.classes_[probs.argmax()]
        ml_conf = probs.max()
    except:
        ml_pred, ml_conf = None, 0.0

    rule_pred = smart_pattern_fallback(cleaned)
    best_score = 0
    for cat, keywords in categories.items():
        scores = [fuzz.partial_ratio(word, cleaned) for word in keywords]
        strong_scores = [s for s in scores if s > 80]
        score = sum(strong_scores) * len(strong_scores)
        if cat == rule_pred:
            best_score = score
            break
    rule_conf = min(best_score / 1000, 1.0)

    if ml_pred == rule_pred:
        return ml_pred
    elif rule_conf > 0.8:
        return rule_pred
    elif ml_conf >= 0.75:
        return ml_pred
    else:
        return rule_pred

df["predicted"] = df["text"].apply(classify_email)
df["correct"] = df["predicted"] == df["category"]
accuracy = df["correct"].mean()

print("\nConfusion Matrix:\n")
print(pd.crosstab(df["category"], df["predicted"]))
print(f"\nOverall Accuracy: {accuracy:.2%}")
