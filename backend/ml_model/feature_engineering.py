# --- START OF FILE: feature_engineering.py ---

import re
import numpy as np
import pandas as pd
from textblob import TextBlob

# This list is the single source of truth for all structured features.
# It defines the features AND their order.
STRUCTURED_FEATURE_COLUMNS = [
    'email_length',
    'num_questions',
    'num_exclamations',
    'contains_table',
    'has_po',
    'has_gst',
    'has_model',
    'has_urgent',
    'has_project',
    'has_specs',
    'has_legal',
    'strong_complaint_hits',
    'has_negative_tone',
    'has_pricing_word',
    'has_certification_request',
    # Subject features are now part of the main list
    'subject_has_quote',
    'subject_has_quotation',
    'subject_has_product',
    'subject_has_price'
]

def extract_subject_features(subject: str) -> list:
    """Extracts features from the email subject line."""
    subject = subject.lower()
    return [
        int("quote" in subject),
        int("quotation" in subject),
        int("product" in subject),
        int("price" in subject),
    ]

def add_features_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame and adds all structured feature columns to it.
    This is used by the TRAINING script.
    """
    # Ensure text column exists and is clean
    df['text'] = df['text'].astype(str).str.lower().str.strip()
    
    # Handle optional subject column
    if 'subject' not in df.columns:
        df['subject'] = "" # Add an empty subject if it's missing
    df['subject'] = df['subject'].astype(str).str.lower().str.strip()

    # --- Text Body Features ---
    df['email_length'] = df['text'].apply(len)
    df['num_questions'] = df['text'].apply(lambda x: x.count('?'))
    df['num_exclamations'] = df['text'].apply(lambda x: x.count('!'))
    df['contains_table'] = df['text'].apply(lambda x: int('description' in x and 'qty' in x or ':' in x))
    df['has_po'] = df['text'].apply(lambda x: 1 if re.search(r'(po[\s#:/\-]*\d+|purchase\s*order)', x, re.IGNORECASE) else 0)
    df['has_gst'] = df['text'].apply(lambda x: 1 if re.search(r'\b\d{2}[a-z]{5}\d{4}[a-z]{1}[a-z\d]{1}z[a-z\d]{1}\b', x, re.IGNORECASE) else 0)
    df['has_model'] = df['text'].apply(lambda x: 1 if re.search(r'model[\s:]*[a-z0-9\-]+', x, re.IGNORECASE) else 0)
    df['has_urgent'] = df['text'].apply(lambda x: 1 if re.search(r'\burgen(t|cy)|immediate(ly)?|asap\b', x, re.IGNORECASE) else 0)
    df['has_project'] = df['text'].apply(lambda x: 1 if re.search(r'\bproject\b.*\b(name|code|id|#)\b', x, re.IGNORECASE) else 0)
    df['has_specs'] = df['text'].apply(lambda x: 1 if re.search(r'\b(specification|technical\s*details?|requirements?)\b', x, re.IGNORECASE) else 0)
    df['has_legal'] = df['text'].apply(lambda x: 1 if re.search(r'\b(terms\s*&?\s*conditions|contract|agreement|warranty)\b', x, re.IGNORECASE) else 0)
    df['strong_complaint_hits'] = df['text'].apply(lambda x: sum(1 for phrase in ["unacceptable", "angry", "disappointed", "fed up", "very poor"] if phrase in x))
    df['has_negative_tone'] = df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity < -0.5 else 0)
    
    pricing_keywords = ["price", "cost", "rate", "quotation", "quote", "charges"]
    df["has_pricing_word"] = df["text"].apply(lambda x: int(any(word in x for word in pricing_keywords)))

    cert_keywords = ["certification", "certified", "isi", "ce", "ul", "approval", "license", "standard", "compliance"]
    df["has_certification_request"] = df["text"].apply(lambda x: int(any(word in x for word in cert_keywords)))

    # --- Subject Features ---
    subject_feats = df['subject'].apply(extract_subject_features).apply(pd.Series)
    subject_feats.columns = [
        'subject_has_quote', 'subject_has_quotation',
        'subject_has_product', 'subject_has_price'
    ]
    df = pd.concat([df, subject_feats], axis=1)

    return df


def create_structured_features_from_text(email_text: str, subject_text: str) -> np.ndarray:
    """
    Creates a numpy array of structured features from raw text.
    The order of features is guaranteed by STRUCTURED_FEATURE_COLUMNS.
    This is used by the APPLICATION script.
    """
    text = email_text.lower()
    
    # Create a temporary DataFrame to reuse the logic
    temp_df = pd.DataFrame([{
        'text': text,
        'subject': subject_text
    }])
    
    # Apply the same feature generation function used for training
    featured_df = add_features_to_df(temp_df)
    
    # Extract the features in the correct order and convert to a numpy array
    feature_values = featured_df[STRUCTURED_FEATURE_COLUMNS].values
    
    # The result is a 2D array, e.g., [[feature1, feature2, ...]]. We need the first (and only) row.
    return feature_values[0]

# --- END OF FILE: feature_engineering.py ---