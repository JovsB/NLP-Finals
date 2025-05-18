import re
import numpy as np
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\bprice\b', 'prize', text)
        return text

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
        has_prize = X_str.str.contains(r'\bprize\b', case=False).astype(int).values.reshape(-1, 1)
        has_selected = X_str.str.contains(r'\bselected\b', case=False).astype(int).values.reshape(-1, 1)
        has_cash = X_str.str.contains(r'\bcash\b', case=False).astype(int).values.reshape(-1, 1)
        has_reward = X_str.str.contains(r'\breward\b', case=False).astype(int).values.reshape(-1, 1)
        has_won = X_str.str.contains(r'\bwon\b', case=False).astype(int).values.reshape(-1, 1)
        has_claim = X_str.str.contains(r'\bclaim\b', case=False).astype(int).values.reshape(-1, 1)
        has_email_address = X_str.str.contains(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', case=False).astype(int).values.reshape(-1, 1)
        has_email_word = X_str.str.contains(r'\bemail\b', case=False).astype(int).values.reshape(-1, 1)

        return np.hstack([
            exclaim_count, has_free, has_url, digit_count, uppercase_ratio,
            has_prize, has_reward, has_selected, has_cash, has_won, has_claim,
            has_email_address, has_email_word
        ])

# --- Define prediction function ---
def predict_spam(texts, pipeline, threshold=0.2):
    if isinstance(texts, str):
        texts = [texts]
    probas = pipeline.predict_proba(texts)[:, 1]
    results = []
    for text, proba in zip(texts, probas):
        # Rule-based override for business scam patterns
        if (
            re.search(r'business opportunity', text, re.IGNORECASE)
            or re.search(r'dear friend', text, re.IGNORECASE)
            or re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)
        ):
            results.append({
                "text": text,
                "spam_probability": max(proba, 0.8),  # force high probability
                "prediction": "spam"
            })
        else:
            results.append({
                "text": text,
                "spam_probability": round(proba, 4),
                "prediction": "spam" if proba >= threshold else "not spam",
                "is_spam": proba >= threshold
            })
    return results

if __name__ == '__main__':
    # --- Load pipeline ---
    print("Loading pipeline...")
    pipeline = joblib.load('models/full_spam_pipeline.joblib')
    
    #sample
    sample_texts = [
        "Dear friend, I have a great business opportunity for you. email:dadasd ",
        "Hi John, just checking in to see how you're doing. ajohn@gmail.com",
        # "Claim your FREE reward by clicking this link!",
        # "I'll be late to the meeting. Traffic is heavy.",
        # "You have been selected for a cash prize. Act fast!",
        # "Full discounts available! Go to https://hellowednesday.io to learn more.",
        # "URGENT: You have been SELECTED to win a PRIZE of $5000!",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2023. Text FA to 87121 to receive entry question(std txt rate)",
    ]

 
    results = predict_spam(sample_texts, pipeline)

    print("\n--- Spam Prediction Results ---")
    for res in results:
        print(f"\nMessage: {res['text']}")
        print(f"→ Spam Probability: {res['spam_probability']}")
        print(f"→ Predicted Label: {res['prediction']}")