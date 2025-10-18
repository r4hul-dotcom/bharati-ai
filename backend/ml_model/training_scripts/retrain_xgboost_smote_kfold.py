# --- START OF FILE: retrain_xgboost_smote_kfold.py ---

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

# NEW: We import our "Instruction Book" to ensure consistency!
from feature_engineering import add_features_to_df, STRUCTURED_FEATURE_COLUMNS

warnings.filterwarnings("ignore")

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

# DELETED: The old, inconsistent add_features function is gone.

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), stop_words='english', min_df=3, max_features=8000,
        sublinear_tf=True, analyzer='word', token_pattern=r'(?u)\b[a-z][a-z0-9_]{2,}\b', max_df=0.7,
    )
    X_tfidf = vectorizer.fit_transform(texts)
    if isinstance(X_tfidf, coo_matrix):
        X_tfidf = X_tfidf.tocsr()
    return X_tfidf, vectorizer

def run_kfold_evaluation_and_train_final_model(X, y, label_encoder, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds = []
    all_targets = []
    print(f"\n--- 🚀 Starting {n_splits}-Fold Cross-Validation ---")
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        print("Applying SMOTE to the training data for this fold...")
        k_neighbors_smote = min(5, min(Counter(y_train).values()) - 1)
        if k_neighbors_smote < 1:
            print("Skipping SMOTE for this fold, not enough samples in the smallest class.")
            X_train_res, y_train_res = X_train, y_train
        else:
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE:", Counter(y_train_res))
        model = XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_estimators=300,
            max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
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
    print(pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_).round(2))
    print(f"\n✅ Overall Accuracy: {accuracy_score(all_targets, all_preds):.3f}")
    print("\n\n--- 🚀 Training Final Model on ALL Data ---")
    print("Applying SMOTE to the entire dataset for final model training...")
    k_neighbors_final = min(5, min(Counter(y).values()) - 1)
    if k_neighbors_final < 1:
       print("Warning: not enough samples for SMOTE on final model. Training without it.")
       X_resampled_final, y_resampled_final = X, y
    else:
        sm_final = SMOTE(random_state=42, k_neighbors=k_neighbors_final)
        X_resampled_final, y_resampled_final = sm_final.fit_resample(X, y)
    print("Final training data class distribution:", Counter(y_resampled_final))
    final_model_base = XGBClassifier(
        objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_estimators=300,
        max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=1.5, use_label_encoder=False
    )
    final_model_calibrated = CalibratedClassifierCV(final_model_base, method='sigmoid', cv=3)
    final_model_calibrated.fit(X_resampled_final, y_resampled_final)
    print("\n✅ Final model has been trained on the full dataset.")
    return final_model_calibrated

def main():
    print("🚀 Starting Enhanced XGBoost Email Classifier Training with K-Fold CV...")
    df = load_data("training_dataset_AUGMENTED_CLEAN.csv")
    
    # CHANGED: We now call our new, consistent function from the instruction book.
    df = add_features_to_df(df)
    
    X_tfidf, vectorizer = vectorize_text(df["text"])
    
    # CHANGED: We get the list of features directly from our instruction book.
    # This guarantees the features and their order are always correct.
    structured = df[STRUCTURED_FEATURE_COLUMNS].values
    
    X_combined = hstack([X_tfidf, structured]).tocsr()
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["category"])
    
    final_model = run_kfold_evaluation_and_train_final_model(X_combined, y_encoded, label_encoder)
    
    # The name of the model is now just 'xgboost_email_classifier.joblib'
    joblib.dump(label_encoder, "label_encoder.joblib")
    joblib.dump(final_model, "xgboost_email_classifier.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    
    print("\n\n--- 🎉 All Artifacts Saved Successfully! ---")
    print("✅ Final Model saved as xgboost_email_classifier.joblib")
    print("✅ Vectorizer saved as tfidf_vectorizer.joblib")
    print("✅ Label encoder saved as label_encoder.joblib")

if __name__ == "__main__":
    main()
# --- END OF FILE: retrain_xgboost_smote_kfold.py ---