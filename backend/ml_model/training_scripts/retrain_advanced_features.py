import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, coo_matrix
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
from textblob import TextBlob
from sklearn.calibration import CalibratedClassifierCV
import nltk
from gensim.models import Word2Vec

# Download necessary NLTK data (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

warnings.filterwarnings("ignore")

# ========== STEP 1: Load Data (No changes) ==========
def load_data(csv_path="training_dataset_enhanced.csv"):
    df = pd.read_csv(csv_path, encoding='utf-8', quotechar='"')
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"].str.lower() != "category"]
    df = df[df["category"] != "feedback"]
    print("\n📊 Initial Class Distribution:")
    print(df["category"].value_counts())
    return df

# ========== STEP 2: Enhanced Feature Engineering (No changes) ==========
def add_features(df):
    df['email_length'] = df['text'].apply(len)
    df['num_questions'] = df['text'].apply(lambda x: x.count('?'))
    df['num_exclamations'] = df['text'].apply(lambda x: x.count('!'))
    df['contains_table'] = df['text'].apply(lambda x: int('Description' in x and 'Qty' in x or ':' in x))
    df['has_po'] = df['text'].apply(lambda x: 1 if re.search(r'(PO[\s#:/\-]*\d+|purchase\s*order)', x, re.IGNORECASE) else 0)
    df['has_gst'] = df['text'].apply(lambda x: 1 if re.search(r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b', x) else 0)
    df['has_model'] = df['text'].apply(lambda x: 1 if re.search(r'model[\s:]*[A-Z0-9\-]+', x, re.IGNORECASE) else 0)
    df['has_urgent'] = df['text'].apply(lambda x: 1 if re.search(r'\burgen(t|cy)|immediate(ly)?|asap\b', x, re.IGNORECASE) else 0)
    df['has_project'] = df['text'].apply(lambda x: 1 if re.search(r'\bproject\b.*\b(name|code|id|#)\b', x, re.IGNORECASE) else 0)
    df['has_specs'] = df['text'].apply(lambda x: 1 if re.search(r'\b(specification|technical\s*details?|requirements?)\b', x, re.IGNORECASE) else 0)
    df['has_legal'] = df['text'].apply(lambda x: 1 if re.search(r'\b(terms\s*&?\s*conditions|contract|agreement|warranty)\b', x, re.IGNORECASE) else 0)
    pricing_keywords = ["price", "cost", "rate", "quotation", "quote", "charges"]
    df["has_pricing_word"] = df["text"].str.lower().apply(lambda x: int(any(word in x for word in pricing_keywords)))
    cert_keywords = ["certification", "certified", "isi", "ce", "ul", "approval", "license", "standard", "compliance"]
    df["has_certification_request"] = df["text"].str.lower().apply(lambda x: int(any(word in x for word in cert_keywords)))
    df['strong_complaint_hits'] = df['text'].apply(lambda x: sum(1 for phrase in ["unacceptable", "angry", "disappointed", "fed up", "very poor"] if phrase in x.lower()))
    df['has_negative_tone'] = df['text'].apply(lambda x: 1 if TextBlob(x).sentiment.polarity < -0.5 else 0)
    # Add this inside your add_features function
    follow_up_phrases = [
    "following up", "any update", "gentle reminder", "checking in", 
    "any news on", "follow-up on", "wanted to check"
    ]
    df["follow_up_signal"] = df["text"].str.lower().apply(
    lambda x: int(any(phrase in x for phrase in follow_up_phrases))
    )
    
    return df

# ========== STEP 3: Vectorization (TF-IDF part remains) ==========
def vectorize_text_tfidf(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), stop_words='english', min_df=3, max_features=5000,
        sublinear_tf=True, analyzer='word', token_pattern=r'(?u)\b[a-z][a-z0-9_]{2,}\b', max_df=0.7,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    if isinstance(X_tfidf, coo_matrix):
        X_tfidf = X_tfidf.tocsr()
    return X_tfidf, vectorizer

# ========== NEW: Word2Vec Feature Generation ==========
def preprocess_for_w2v(texts):
    """Tokenize and clean text for Word2Vec training."""
    # Simple tokenizer using NLTK
    tokenized_texts = [nltk.word_tokenize(text.lower()) for text in texts]
    return tokenized_texts

def create_document_vectors(tokenized_texts, w2v_model):
    """Average word vectors for each document."""
    document_vectors = []
    vector_size = w2v_model.vector_size
    for tokens in tokenized_texts:
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if len(vectors) > 0:
            document_vectors.append(np.mean(vectors, axis=0))
        else:
            # If no words in the document are in the vocab, add a vector of zeros
            document_vectors.append(np.zeros(vector_size))
    return np.array(document_vectors)

# ========== STEP 4: K-Fold Evaluation (Updated to handle new features) ==========
def run_kfold_evaluation_and_train_final_model(X_combined, y, label_encoder, n_splits=5):
    """
    Performs Stratified K-Fold cross-validation for robust evaluation,
    then trains and returns a final model on the entire dataset.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds, all_targets = [], []
    
    print(f"\n--- 🚀 Starting {n_splits}-Fold Cross-Validation with Advanced Features ---")
    
    for fold, (train_index, val_index) in enumerate(skf.split(X_combined, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        X_train, X_val = X_combined[train_index], X_combined[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        print("Applying SMOTE to the training data for this fold...")
        k_neighbors = min(5, min(Counter(y_train).values()) - 1)
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE:", Counter(y_train_res))

        model = XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_estimators=350,
            max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.5, use_label_encoder=False
        )
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        calibrated_model.fit(X_train_res, y_train_res)
        
        y_pred = calibrated_model.predict(X_val)
        all_preds.extend(y_pred)
        all_targets.extend(y_val)

    print("\n\n--- ✅ Overall Cross-Validation Results ---")
    print("\n📊 Aggregated Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=label_encoder.classes_, zero_division=0))

    print("\n🧩 Aggregated Confusion Matrix (Normalized):")
    cm = confusion_matrix(all_targets, all_preds, normalize='true')
    print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).round(3))
    
    print(f"\n✅ Overall Accuracy: {accuracy_score(all_targets, all_preds):.4f}")

    print("\n\n--- 🚀 Training Final Model on ALL Data ---")
    print("Applying SMOTE to the entire dataset for final model training...")
    k_neighbors_final = min(5, min(Counter(y).values()) - 1)
    sm_final = SMOTE(random_state=42, k_neighbors=k_neighbors_final)
    X_resampled_final, y_resampled_final = sm_final.fit_resample(X_combined, y)

    final_model_base = XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_estimators=350,
        max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.5, use_label_encoder=False
    )
    final_model_calibrated = CalibratedClassifierCV(final_model_base, method='sigmoid', cv=3)
    final_model_calibrated.fit(X_resampled_final, y_resampled_final)
    
    print("\n✅ Final model has been trained on the full dataset.")
    return final_model_calibrated

# ========== STEP 5: Main Training Pipeline (HEAVILY REVISED) ==========
def main():
    print("🚀 Starting Advanced XGBoost Classifier Training (TF-IDF + Word2Vec)...")
    df = load_data("training_dataset_enhanced.csv")
    
    # --- Feature Set 1: TF-IDF ---
    print("\n[1/3] Generating TF-IDF features...")
    X_tfidf, vectorizer = vectorize_text_tfidf(df["text"])
    print(f"TF-IDF feature shape: {X_tfidf.shape}")
    
    # --- Feature Set 2: Word2Vec ---
    print("\n[2/3] Generating Word2Vec features...")
    # Preprocess text for Word2Vec
    tokenized_texts = preprocess_for_w2v(df["text"])
    # Train a custom Word2Vec model
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)
    w2v_model.save("word2vec_model.bin")
    # Create document vectors by averaging word vectors
    X_word2vec = create_document_vectors(tokenized_texts, w2v_model)
    print(f"Word2Vec feature shape: {X_word2vec.shape}")

    # --- Feature Set 3: Manual Structured Features ---
    print("\n[3/3] Loading manual structured features...")
    structured_features = [
        'email_length', 'num_questions', 'num_exclamations', 'contains_table', 
        'has_po', 'has_gst', 'has_model', 'has_urgent', 'has_project', 
        'has_specs', 'has_legal', 'has_pricing_word', 'has_certification_request',
        'follow_up_signal', 'is_escalation', 'context_reference'
    ]
    structured = df[structured_features].values
    print(f"Structured feature shape: {structured.shape}")
    
    # --- Combine all feature sets ---
    print("\nCombining all feature sets...")
    # hstack works with a mix of sparse (TF-IDF) and dense (W2V, Structured) arrays
    X_combined = hstack([X_tfidf, X_word2vec, structured]).tocsr()
    print(f"Combined feature shape: {X_combined.shape}")
    
    # --- Target Variable ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["category"])
    
    # --- Run CV evaluation and train the final model ---
    final_model = run_kfold_evaluation_and_train_final_model(X_combined, y_encoded, label_encoder)
    
    # --- Save all artifacts ---
    joblib.dump(label_encoder, "label_encoder_advanced.joblib")
    joblib.dump(final_model, "xgboost_classifier_advanced.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer_advanced.joblib")
    # The Word2Vec model is already saved above
    
    print("\n\n--- 🎉 All Advanced Artifacts Saved Successfully! ---")
    print("✅ Final Model saved as xgboost_classifier_advanced.joblib")
    print("✅ Vectorizer saved as tfidf_vectorizer_advanced.joblib")
    print("✅ Label encoder saved as label_encoder_advanced.joblib")
    print("✅ Word2Vec model saved as word2vec_model.bin")

if __name__ == "__main__":
    main()