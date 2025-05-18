import re
import numpy as np
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# NOTE: Custom transformers must EXACTLY match training implementation 

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        return X.astype(str).apply(self.clean_text)

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'http\S+|www\.\S+', ' urlplaceholder ', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

class NumericFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        X_str = X.astype(str)

        exclaim_count = X_str.str.count('!').values.reshape(-1, 1)
        has_free = X_str.str.contains(r'\bfree\b', case=False).astype(int).values.reshape(-1, 1)
        has_url = X_str.str.contains(r'http\S+|www\.\S+', case=False).astype(int).values.reshape(-1, 1)
        digit_count = X_str.str.count(r'\d').values.reshape(-1, 1)
        uppercase_ratio = X_str.apply(
            lambda s: sum(1 for c in s if c.isupper()) / len(s) if len(s) > 0 else 0
        ).values.reshape(-1, 1)

        return np.hstack([exclaim_count, has_free, has_url, digit_count, uppercase_ratio])

#  Load pipeline---
PIPELINE_PATH = 'models/full_spam_pipeline.joblib'
THRESHOLD = 0.6

print("Loading pipeline...")
pipeline = joblib.load(PIPELINE_PATH)

#  Prediction function 
def predict_spam(texts, pipeline, threshold=THRESHOLD):
    if isinstance(texts, str):
        texts = [texts]
    probas = pipeline.predict_proba(texts)[:, 1]
    return [{
        "text": text,
        "spam_probability": round(proba, 4),
        "prediction": "spam" if proba >= threshold else "not spam"
    } for text, proba in zip(texts, probas)]

#  Sample messages / feel free try different mssgs
sample_texts = [
    "Congratulations! You won a free ticket to Bahamas! Call now!",
    "Hi John, just checking in to see how you're doing.",
    "Claim your FREE reward by clicking this link!",
    "I'll be late to the meeting. Traffic is heavy.",
    "You have been selected for a cash prize. Act fast!"
    "Full discounts available! Go to https://hellowednesday.io to learn more.",
]

results = predict_spam(sample_texts, pipeline)

print("\n--- Spam Prediction Results ---")
for res in results:
    print(f"\nMessage: {res['text']}")
    print(f"→ Spam Probability: {res['spam_probability']}")
    print(f"→ Predicted Label: {res['prediction']}")