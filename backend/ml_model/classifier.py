import joblib
import json
import nltk
import numpy as np
import string
import traceback
from pathlib import Path
from rapidfuzz import fuzz
from scipy.sparse import hstack
from nltk.tokenize import sent_tokenize
from textblob import TextBlob

# --- IMPORTANT: Import your local feature_engineering file ---
from .feature_engineering import create_structured_features_from_text


# ===================================================================
# === START: LAZY LOADING SETUP ===
# ===================================================================

# 1. Define global variables for resources, but initialize them as empty.
#    This part of the code runs instantly when the app starts, using no time or memory.
_resources = {
    "model": None,
    "vectorizer": None,
    "label_encoder": None,
    "PRODUCT_HIERARCHY": {},
    "PRODUCT_CODE_TO_NAME_MAP": {},
    "resources_loaded": False  # A flag to check if loading has occurred
}

# This is the function that will contain ALL your original model/data loading code.
def _load_resources():
    """
    An internal function to load all ML models and data files into memory.
    This function will only run ONCE, the first time it's needed.
    """
    # Check the flag. If resources are already loaded, do nothing and return instantly.
    if _resources["resources_loaded"]:
        return

    print("--- First request received. Loading all ML models and data files now... ---")
    
    # --- Model Loading Logic (Moved from the top of your original file) ---
    try:
        current_dir = Path(__file__).parent
        model_dir = current_dir / "saved_model"
        
        _resources["model"] = joblib.load(model_dir / "xgboost_email_classifier.joblib")
        _resources["vectorizer"] = joblib.load(model_dir / "tfidf_vectorizer.joblib")
        _resources["label_encoder"] = joblib.load(model_dir / "label_encoder.joblib")
        
        print("✅ All ML components loaded successfully.")

    except Exception as e:
        print(f"🔴 CRITICAL ERROR while loading model components: {e}")
        raise e # Stop the process if models can't be loaded

    # --- Product Hierarchy Loading Logic (Moved from your original file) ---
    try:
        current_dir = Path(__file__).parent
        json_path = current_dir.parent / "data" / "product_data" / "structured_product_hierarchy.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            _resources["PRODUCT_HIERARCHY"] = json.load(f)
        print(f"✅ Product hierarchy loaded successfully.")
        
        # Call the helper to build the product map
        _build_product_map(_resources["PRODUCT_HIERARCHY"])

    except Exception as e:
        print(f"🔴 CRITICAL ERROR while loading product hierarchy: {e}")
        raise e

    # --- Set the flag to True at the end ---
    _resources["resources_loaded"] = True
    print("--- All resources are now loaded and cached in memory. ---")


def _build_product_map(hierarchy_data):
    """
    Helper function to populate the product map inside the central _resources dictionary.
    This is called by the main _load_resources() function.
    """
    # This function now modifies a key within the _resources dictionary,
    # which is better practice than using the 'global' keyword.
    
    top_level_names = {
        "FEXT": "Fire Extinguishers", "MOD": "Modulars", "LITH": "Lith-Ex Safety",
        "AERO": "Aerosol Systems", "DET": "Detectors", "HYD": "Hydrant Systems",
        "FET": "Equipment Trolleys", "SYS": "Suppression Systems"
    }
    
    # Reset the map before building to ensure freshness
    _resources["PRODUCT_CODE_TO_NAME_MAP"] = {}

    def recurse(node, path_display, path_code):
        if isinstance(node, dict):
            for key, value in node.items():
                current_display_part = top_level_names.get(key, key) if not path_display else key
                new_path_display = f"{path_display} > {current_display_part}" if path_display else current_display_part
                current_code_part = key.upper().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                new_path_code = f"{path_code}-{current_code_part}" if path_code else current_code_part
                
                if isinstance(value, list) and not value:
                    # Access and modify the dictionary within _resources
                    _resources["PRODUCT_CODE_TO_NAME_MAP"][new_path_code] = new_path_display
                elif isinstance(value, list) and value:
                     for item in value:
                        if isinstance(item, dict) and 'name' in item and 'code' in item:
                            display_name = f"{new_path_display} > {item['name']}"
                            # Access and modify the dictionary within _resources
                            _resources["PRODUCT_CODE_TO_NAME_MAP"][item['code']] = display_name
                elif isinstance(value, dict):
                    recurse(value, new_path_display, new_path_code)

    # Start the recursive process
    recurse(hierarchy_data, "", "")
    
    # Update the console log
    map_size = len(_resources["PRODUCT_CODE_TO_NAME_MAP"])
    print(f"✅ Product code-to-name map built successfully with {map_size} items.")

# Note: The code that originally loaded the JSON file and called this function
# has already been moved inside the main `_load_resources()` function in my previous answer.
# This helper function, `_build_product_map`, will be called from within there.

# === Global Definitions ===
categories = {
    "quotation_request": [
        "price quote", "send quote", "product quote", "quote request", "quotation required",
        "quotation", "quote", "price", "estimate", "cost", "rate", "pricing",
        "quotation needed", "need quotation", "send price", "price list",
        "cost estimate", "price quote", "request for quote", "rfq",
        "how much for", "what's the price of", "looking for pricing on",
        "please quote", "RFQ", "could you provide a quote", "price inquiry",
        "requesting quotation", "need pricing", "require quotation",
        "send me the quote", "what would be the cost", "price details",
        "quote request", "requesting price", "cost breakdown",
        "please send quotation", "need cost estimate", "pricing information"
    ],
    "complaint": [
        "not functioning", "doesn't work", "does not function", "unit damaged", "received broken", "arrange replacement", "not operational",
        "complaint", "complaints", "problem", "problems", "issue", "issues",
        "damaged", "broken", "not working", "faulty", "defective",
        "disappointed", "dissatisfied", "bad product", "poor quality",
        "does not work", "malfunction", "refund", "return", "replacement",
        "missing item", "not delivered", "wrong item", "incorrect item",
        "delivered wrong", "delivered less", "short delivery",
        "quantity mismatch", "wrong delivery", "defective unit",
        "urgent resolution", "received only", "invoice mismatch",
        "not as described", "poor service", "unhappy with",
        "this is unacceptable", "very disappointed", "not satisfied",
        "quality issue", "service complaint", "shipping problem",
        "package damaged", "item broken", "not what I ordered",
        "order mistake", "billing error", "overcharged", "wrong billing",
        "need immediate resolution", "compensation required",
        "want my money back", "how do I return this", "not up to standard"
    ],
    "follow_up": [
        "follow up", "follow-up", "followup", "reminder", "reminders",
        "status", "status update", "pending", "update", "any update",
        "awaiting", "waiting", "still waiting", "revert", "response",
        "haven't heard back", "no reply yet", "when can I expect",
        "please respond", "kindly update", "need update",
        "requesting status", "what's the status", "any progress",
        "could you update me", "following up on", "checking in on",
        "when will this be", "has this been", "did you receive",
        "confirm receipt", "please acknowledge", "awaiting your reply",
        "looking forward to", "please let me know", "need confirmation",
        "requesting follow up", "status check", "progress update,"
        "dispatch", "dispatched", "delivery", "deliver", "courier",
        "couriers", "shipment", "shipping", "track", "tracking",
        "lr copy", "lr number", "tracking id", "tracking number",
        "awb", "airway bill", "delivery status", "logistics",
        "transport", "transit", "shipped", "out for delivery",
        "in transit", "on the way", "expected delivery",
        "delivery date", "ship date", "dispatch date",
        "when will it ship", "has it been shipped",
        "where is my order", "order status", "shipment update",
        "delivery timeline", "expected arrival", "arrival date",
        "package location", "where's my package", "track my order",
        "delivery confirmation", "proof of delivery", "pod",
        "received shipment", "not received", "delivery problem",
        "shipping details", "carrier information", "transport details",
        "logistics update", "freight information"
    ],


    "general_enquiry": [
        "enquiry", "inquiry", "enquiries", "inquiries", "information",
        "info", "clarification", "clarify", "details", "detail",
        "need details", "require info", "want to know", "would like to know",
        "kindly confirm", "please confirm", "product details",
        "share catalog", "send brochure", "what is the", "can you explain",
        "please share specs", "technical specifications", "any brochure",
        "looking for", "seeking information", "technical sheet",
        "need data", "require details", "what type of",
        "how does it work", "how to use", "training available",
        "can you guide", "service schedule", "provide availability",
        "have questions", "need assistance", "require clarification",
        "more information", "additional details", "further information",
        "could you explain", "would like information", "seeking clarification",
        "please advise", "need guidance", "require explanation",
        "product information", "service information", "company information",
        "contact information", "pricing information", "availability information",
        "specification request", "feature inquiry", "capacity details",
        "dimension inquiry", "material information", "operation details",
        "maintenance inquiry", "installation query", "warranty information"
    ],

    "other": [
        # --- Newsletters & Promotions ---
        "newsletter", "subscription", "subscribe", "unsubscribe", "view in browser",
        "limited time offer", "special offer", "deal of the day", "flash sale",
        "discount", "coupon code", "promo code", "shop now", "view collection",
        "new arrivals", "bestsellers", "don't miss out", "last chance",
        "our blog", "read more", "latest news", "weekly update",
        
        # --- Account & Security Alerts ---
        "security alert", "your account", "verify your email", "confirm your account",
        "password reset", "your password was changed", "new sign-in", "login attempt",
        "account details", "privacy policy", "terms of service", "update your profile",
        "2-factor authentication", "one-time password", "otp",
        
        # --- Social Media & Notifications ---
        "mentioned you on", "new connection request", "someone liked your post",
        "invites you to connect", "from your network", "new message from",
        "friend request", "new follower", "LinkedIn", "Facebook", "Twitter",
        
        # --- Generic & Spammy ---
        "congratulations you've won", "claim your prize", "you have been selected",
        "work from home",
        "make money fast", "viagra", "cialis", "weight loss", "get a loan",
        "no longer wish to receive", "if you believe this is in error", "click here"
    ]
}


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def detect_intent(text):
    intent_rules = {
        "action_required": ["please", "kindly", "request", "need", "require", "awaiting", "approve", "follow up", "urgent"],
        "information_only": ["for your information", "fyi", "note", "just informing", "letting you know"],
        "confirmation": ["we confirm", "confirmed", "acknowledge", "we acknowledge", "as discussed", "as per our conversation"],
        "query": ["can you", "could you", "would you", "may i know", "how much", "when is", "what is", "do you have", "share the details"],
        "appreciation": ["thank you", "thanks", "appreciate", "grateful", "great job", "good work", "kudos", "well done"]
    }
    text = text.lower()
    scores = {intent: sum(1 for phrase in phrases if phrase in text) for intent, phrases in intent_rules.items()}
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_intents[0][0] if sorted_intents[0][1] > 0 else "unknown"

def detect_urgency(email_text):
    """Detect urgency level from email text (0-1 scale)"""
    email_text = email_text.lower()
    urgency_score = 0.0
    
    # Strong urgency indicators
    if any(word in email_text for word in ["urgent", "immediately", "asap", "right away", "emergency"]):
        urgency_score = 0.9
    # Moderate urgency
    elif any(word in email_text for word in ["soon", "quick", "prompt", "timely", "follow up"]):
        urgency_score = 0.6
    # Mild urgency
    elif any(word in email_text for word in ["when convenient", "at your earliest", "please respond"]):
        urgency_score = 0.3
    
    return min(max(urgency_score, 0), 1)  # Ensure between 0-1



def smart_pattern_fallback(text):
    text = text.lower().strip()

    # The patterns dictionary now includes our comprehensive "other" category
    patterns = {
        "quotation_request": [
            "price quote", "send quote", "product quote", "quote request", "quotation required",
            "please send quotation", "need a quote", "request for quotation", "provide the rates", 
            "price list", "send cost", "need price", "quote us", "share pricing", "quotation for", 
            "rfq", "rate contract", "commercial offer", "could you quote", "price estimate",
            "looking for pricing", "what's your rate", "cost breakdown", "requesting quote",
            "please provide quote", "need your rates", "seeking quotation", "price inquiry",
            "request for pricing", "send me the quote", "what would be the cost", "require quotation",
            "share your commercial proposal", "send me your offer", "budgetary offer", "commercial quotation"
        ],
        "complaint": [
            "not functioning", "doesn't work", "does not function", "unit damaged", "received broken", "arrange replacement", "not operational",
            "not working", "damaged", "leaking", "defective", "faulty", "not satisfied", 
            "wrong item", "complaint", "need replacement", "issue with", "return this", 
            "broken", "malfunction", "poor quality", "disappointed", "unhappy with",
            "not as described", "received damaged", "defect in", "this is unacceptable", "want refund", 
            "return policy", "compensation", "service issue", "shipping problem", 
            "missing parts", "incomplete delivery", "billing error", "overcharged", 
            "never received", "item missing", "how to return", "warranty claim", 
            "dissatisfied with", "very disappointed"
        ],
        "follow_up": [
            "just following up", "any update", "still waiting", "haven't heard", 
            "reminder", "status of our previous mail", "awaiting your reply", 
            "follow up", "please revert", "checking status", "pending since",
            "when can I expect", "no response yet", "kindly update", "need update",
            "requesting status", "what's the status", "following up on",
            "please respond", "awaiting confirmation", "has this been processed",
            "did you receive my", "could you update", "looking forward to",
            "need your response", "please acknowledge", "reminder about",
            "status check", "progress update", "when will this be", "per our call", 
            "revised proposal", "waiting for proposal", "awaiting document",
            "dispatched", "tracking id", "awb number", "courier", "lr copy", 
            "shipment", "delivery update", "logistics", "delivery status", 
            "dispatch details", "shipping confirmation", "out for delivery",
            "track my order", "where is my", "expected delivery date",
            "shipment tracking", "transport details", "in transit",
            "when will it arrive", "has it been shipped", "delivery timeline",
            "proof of delivery", "pod", "freight details", "carrier information",
            "logistics update", "package location", "transit status",
            "ship date", "dispatch confirmation", "order tracking"
        ],
        "general_enquiry": [
            "enquiry", "want information", "details about", "can you share", 
            "product specification", "availability of", "lead time", "need info", 
            "clarification", "looking for details", "require information",
            "could you explain", "would like to know", "please provide details",
            "need assistance with", "have a question about", "seeking clarification",
            "more information about", "technical specifications", "product features",
            "how does it work", "what are the options", "service details",
            "company information", "contact details", "catalog request",
            "brochure needed", "spec sheet", "operation manual",
            "installation guide", "maintenance information", "warranty details",
            "can you confirm price", "clarify pricing", "what is the cost",
            "need clarification on pricing"
        ],
        "other": [
            "newsletter", "subscription", "subscribe", "unsubscribe", "view in browser",
            "limited time offer", "special offer", "deal of the day", "flash sale",
            "discount", "coupon code", "promo code", "shop now", "view collection",
            "security alert", "your account", "verify your email", "confirm your account",
            "password reset", "your password was changed", "new sign-in", "login attempt",
            "mentioned you on", "new connection request", "congratulations you've won"
        ]
    }

    # --- START: NEW PRIORITIZED LOGIC ---

    # Step 1: Check for strong "other" signals FIRST.
    # If we find one, we can be confident it's not a business email and stop early.
    for phrase in patterns["other"]:
        if phrase in text:
            return "other"

    # Step 2: If it's not "other", then proceed with the business logic.
    # Price intent disambiguation is still useful here.
    if "price" in text or "cost" in text:
        if any(kw in text for kw in ["quote", "quotation", "send", "provide", "share", "need", "request", "offer"]):
            return "quotation_request"
        elif any(kw in text for kw in ["what is", "can you", "confirm", "clarify", "is it", "how much"]):
            return "general_enquiry"

    # Step 3: Loop through the main business categories.
    for category, phrases in patterns.items():
        if category == "other":  # We've already checked this
            continue
        for phrase in phrases:
            if phrase in text:
                return category
    
    # --- END: NEW PRIORITIZED LOGIC ---
    
    # If no pattern is found at all, return None. The calling function will handle it.
    return None

def normalize(text):
    """Lowercase and simplify text for matching."""
    if isinstance(text, dict):  # Skip dictionaries
        return ""
    return str(text).lower().replace(" ", "").replace("-", "").replace("_", "")


def detect_products(text: str) -> list:
    """
    Detects products in text by matching keywords from the hierarchy
    and returns their official product codes. This version can detect
    both final products and selectable categories.
    """
    # === LAZY LOADING TRIGGER AND DATA ACCESS ===
    # ADDED: This line ensures that all models and data files are loaded before proceeding.
    _load_resources()
    # CHANGED: Get the hierarchy data from our central, loaded resources dictionary.
    PRODUCT_HIERARCHY = _resources["PRODUCT_HIERARCHY"]

    # --- The rest of your original logic remains exactly the same ---
    detected_codes = []
    clean_text = normalize(text)

    def recurse_hierarchy(node, path_code):
        if isinstance(node, dict):
            for key, value in node.items():
                current_code_part = key.upper().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                new_path_code = f"{path_code}-{current_code_part}" if path_code else current_code_part

                # Match against the category name itself
                if normalize(key) in clean_text:
                    if isinstance(value, list) and not value:
                        # This is a selectable category like "Smoke Detectors"
                        detected_codes.append(new_path_code)

                # Process final products in a list
                if isinstance(value, list) and value:
                    for item in value:
                        if isinstance(item, dict) and 'name' in item and 'code' in item:
                            if normalize(item['name']) in clean_text:
                                detected_codes.append(item['code'])
                # Go deeper into the hierarchy
                elif isinstance(value, dict):
                    recurse_hierarchy(value, new_path_code)

    # This line now correctly uses the PRODUCT_HIERARCHY variable we defined at the top.
    recurse_hierarchy(PRODUCT_HIERARCHY, "")
    return list(set(detected_codes))


def extract_subject_features(subject):
    subject = subject.lower()
    return [
        int("quote" in subject),
        int("quotation" in subject),
        int("product" in subject),
        int("price" in subject),
    ]


def sentence_level_fallback(text):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split(".")

    votes = {}

    # Internal fallback disambiguation (price-specific)
    price_phrases = ["price", "cost"]
    quote_verbs = ["quote", "quotation", "send", "provide", "share", "need", "request", "offer"]
    enquiry_verbs = ["what is", "can you", "confirm", "clarify", "is it", "how much"]

    for sentence in sentences:
        sentence = sentence.lower().strip()

        # Disambiguate pricing intent per sentence (independent of smart_pattern_fallback)
        if any(p in sentence for p in price_phrases):
            if any(v in sentence for v in quote_verbs):
                votes["quotation_request"] = votes.get("quotation_request", 0) + 1
                continue
            elif any(v in sentence for v in enquiry_verbs):
                votes["general_enquiry"] = votes.get("general_enquiry", 0) + 1
                continue

        # Standard category voting
        cat = smart_pattern_fallback(sentence)
        if cat:
            votes[cat] = votes.get(cat, 0) + 1

    # Return category with most votes, default to general_enquiry
    return max(votes.items(), key=lambda x: x[1])[0] if votes else "general_enquiry"




def classify_email(email_text, subject_text=""):
    """
    Classifies an email by combining a machine learning model with a rule-based system.
    This function now triggers the loading of models and data on its first run.
    """
    # === LAZY LOADING TRIGGER ===
    # The very first time this function is called, it will run the resource loader.
    # Every subsequent call will skip this step instantly as the resources will already be in memory.
    _load_resources()
    
    # Now, access all your models and data from the central _resources dictionary.
    # This replaces the old 'global' keyword and direct variable access.
    model = _resources["model"]
    vectorizer = _resources["vectorizer"]
    label_encoder = _resources["label_encoder"]

    # --- Your original classification logic starts here ---
    # This code is copied directly from your file, with no changes needed to the logic itself.
    cleaned = preprocess(email_text)
    products_detected = detect_products(cleaned) # This will now use the loaded product hierarchy

    ml_prediction = None
    ml_confidence = 0.0

    # Get the rule-based prediction first, as a fallback
    rule_prediction = sentence_level_fallback(email_text)
    rule_confidence = 0.3 if rule_prediction else 0.0

    # --- Start of the Machine Learning Prediction block ---
    if model and vectorizer and label_encoder:
        try:
            # 1. Vectorize the email body using the loaded TF-IDF vectorizer.
            X_text = vectorizer.transform([cleaned])

            # 2. Create the structured features using our reliable, imported function.
            structured_features_array = create_structured_features_from_text(cleaned, subject_text)

            # 3. Combine the text features and structured features into the final vector.
            # No need to re-import hstack if it's already at the top of the file
            X_final = hstack([X_text, structured_features_array.reshape(1, -1)])

            # Make the prediction
            proba = model.predict_proba(X_final)[0]
            max_idx = proba.argmax()
            ml_prediction = label_encoder.inverse_transform([max_idx])[0]
            ml_confidence = float(proba[max_idx])

        except Exception as e:
            print(f"🔴 ML prediction failed! Error: {e}")
            print(traceback.format_exc())

    # --- Start of the Hybrid Logic ---
    if rule_prediction:
        rule_confidence = sum([
            fuzz.partial_ratio(cleaned, phrase)
            for phrase in categories.get(rule_prediction, [])
            if fuzz.partial_ratio(cleaned, phrase) > 80
        ]) / 1000

    if 0.3 < ml_confidence < 0.7 and rule_confidence > 0.5:
        print("⚖️ Mid-confidence ML → Rule stronger: Overriding with rule.")
        ml_prediction = rule_prediction
        ml_confidence = rule_confidence
    elif ml_confidence < 0.3 and rule_confidence >= 0.3 and rule_prediction:
        print("🔁 Overriding with sentence-level fallback due to low confidence.")
        ml_prediction = rule_prediction
        ml_confidence = rule_confidence

    scores = {}
    all_categories = list(set([ml_prediction, rule_prediction] + list(categories.keys())))
    for cat in all_categories:
        if cat is None: continue
        cat_score = 0
        if cat == ml_prediction: cat_score += 0.5 * ml_confidence
        if cat == rule_prediction: cat_score += 0.3 * rule_confidence
        
        booster_words = {
            "general_enquiry": ["need information", "catalogue", "specifications", "how does it work"],
            "complaint": ["not working", "damaged", "broken", "missing", "invoice mismatch"],
            "follow_up": ["still waiting", "dispatch", "status update", "revert", "reminder"],
            "quotation_request": ["quote", "quotation", "rfq", "price list", "cost estimate"],
            "other": ["unsubscribe", "security alert", "your order", "password reset", "newsletter"]
        }
        for word in booster_words.get(cat, []):
            if word in cleaned: cat_score += 0.2
        scores[cat] = round(cat_score, 4)

    # --- Booster logic ---
    if "price" in cleaned and any(w in cleaned for w in ["not received", "delayed", "wrong", "overcharged"]):
        print("🧠 Booster: price + issue trigger → boosting complaint")
        scores["complaint"] = scores.get("complaint", 0) + 0.2
    if any(w in cleaned for w in ["price", "quotation", "quoted"]) and any(w in cleaned for w in ["mismatch", "discrepancy"]):
        print("🧠 Booster: price + discrepancy → complaint")
        scores["complaint"] = scores.get("complaint", 0) + 0.3
    if any(w in cleaned for w in ["follow up", "reminder", "awaiting", "haven’t received"]) and "quotation" in cleaned:
        print("🧠 Booster: follow-up on quotation detected")
        scores["follow_up"] = scores.get("follow_up", 0) + 0.3

    # --- Disambiguation Logic ---
    is_likely_follow_up = ml_prediction == 'follow_up' or rule_prediction == 'follow_up'
    has_strong_quote_signal = any(kw in cleaned for kw in ["please quote", "quotation for", "quote for", "send quote", "rfq"])

    if is_likely_follow_up and has_strong_quote_signal:
        print("🧠 Disambiguation: 'follow_up' prediction is likely confused by a quotation request. Adjusting scores.")
        if 'follow_up' in scores:
            scores['follow_up'] -= 0.4
        scores['quotation_request'] = scores.get('quotation_request', 0) + 0.5

    # --- Safer Final Prediction Logic ---
    if scores:
        final_prediction = max(scores.items(), key=lambda x: x[1])[0]
        if max(scores.values()) < 0.2:
            print("📉 All scores are low, defaulting to 'other'.")
            final_prediction = "other"
    else:
        final_prediction = rule_prediction or "other"

    strong_other_keywords = ["security alert", "your account", "invoice overdue", "password", "subscription", "verify your email", "login attempt"]
    
    if final_prediction == 'complaint' and ml_confidence < 0.8:
        if any(keyword in cleaned for keyword in strong_other_keywords):
            print("🧠 Override: 'complaint' prediction looks like a security/admin alert. Changing to 'other'.")
            final_prediction = "other"

    print("🧠 Hybrid Debug Info:")
    print(f"  ML Prediction: {ml_prediction} (Confidence: {round(ml_confidence, 3)})")
    print(f"  Rule Prediction: {rule_prediction} (Confidence: {round(rule_confidence, 3)})")
    print(f"  Hybrid Scores: {scores}")
    print(f"  Final Prediction: {final_prediction}")

    return final_prediction, {
        "ml_prediction": ml_prediction,
        "ml_confidence": float(ml_confidence or 0),
        "rule_prediction": rule_prediction,
        "rule_confidence": round(rule_confidence, 3),
        "hybrid_scores": scores,
        "products_detected": products_detected,
    }