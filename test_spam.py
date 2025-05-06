import re
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# --- Re-declare custom transformers used in the original pipeline ---
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [self.clean_text(text) for text in X]
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
        return re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single one

class NumericFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Ensure X is a list or pandas Series
        if isinstance(X, list):
            X = np.array(X)
        # Count exclamation marks in each text
        exclaim_count = np.array([text.count('!') for text in X]).reshape(-1, 1)
        # Check if the word "free" exists in each text
        has_free = np.array([1 if 'free' in text.lower() else 0 for text in X]).reshape(-1, 1)
        return np.hstack([exclaim_count, has_free])

# --- Load pipeline ---
PIPELINE_PATH = 'models/full_spam_pipeline.joblib'
THRESHOLD = 0.6

print("Loading pipeline...")
pipeline = joblib.load(PIPELINE_PATH)

# --- Define prediction function ---
def predict_spam(texts, pipeline, threshold=THRESHOLD):
    if isinstance(texts, str):
        texts = [texts]   
    probas = pipeline.predict_proba(texts)[:, 1]   
    predictions = []
    for text, proba in zip(texts, probas):
        label = "spam" if proba >= threshold else "not spam"
        predictions.append({
            "text": text,
            "spam_probability": round(proba, 4),
            "prediction": label
        })
    return predictions

 
sample_texts = [
    "Congratulations! You won a free ticket to Bahamas! Call now!",
    "Hi John, just checking in to see how you're doing.",
    "Claim your FREE reward by clicking this link!",
    "I'll be late to the meeting. Traffic is heavy.",
    "You have been selected for a cash prize. Act fast!"
]

results = predict_spam(sample_texts, pipeline)

print("\n--- Spam Prediction Results ---")
for res in results:
    print(f"\nMessage: {res['text']}")
    print(f"→ Spam Probability: {res['spam_probability']}")
    print(f"→ Predicted Label: {res['prediction']}")
