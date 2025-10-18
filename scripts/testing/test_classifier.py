
import pytest
from testapp import smart_pattern_fallback, classify_email

# === Test Cases for smart_pattern_fallback ===

def test_fallback_general_enquiry():
    text = "Can you share the brochure and warranty details?"
    assert smart_pattern_fallback(text) == "general_enquiry"

def test_fallback_complaint_keywords():
    text = "The extinguisher was damaged and not working."
    assert smart_pattern_fallback(text) == "complaint"

def test_fallback_follow_up_delivery():
    text = "Please confirm dispatch status and share tracking ID."
    assert smart_pattern_fallback(text) == "follow_up"

def test_fallback_quotation_request():
    text = "Please send quotation for 5 units of ABC extinguishers."
    assert smart_pattern_fallback(text) == "quotation_request"

def test_fallback_mixed_price_enquiry():
    text = "Can you clarify what the cost includes?"
    assert smart_pattern_fallback(text) == "general_enquiry"

# === Test Cases for classify_email (Requires model & vectorizer loaded) ===

def test_classify_complaint_with_price_issue():
    text = "The quoted price was approved, but the invoice was higher. This is unacceptable."
    pred, debug = classify_email(text)
    assert pred == "complaint"
    assert debug["hybrid_scores"]["complaint"] > 0.5

def test_classify_simple_quote_request():
    text = "Please send your best price for 3 units of CO2 extinguisher."
    pred, debug = classify_email(text)
    assert pred == "quotation_request"

def test_classify_certification_enquiry():
    text = "Do your extinguishers carry ISI and CE certifications?"
    pred, debug = classify_email(text)
    assert pred == "general_enquiry"

def test_classify_follow_up_polite():
    text = "We are still waiting for your dispatch confirmation. Kindly update us."
    pred, debug = classify_email(text)
    assert pred == "follow_up"

def test_classify_ambiguous_quote_with_delay():
    text = "Please send your quote. We haven’t received the previous shipment yet."
    pred, debug = classify_email(text)
    assert "quotation_request" in debug["hybrid_scores"]
    assert "complaint" in debug["hybrid_scores"]

