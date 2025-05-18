import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

DATA_PATH = 'dataset/spam.csv'
PIPELINE_PATH = 'models/full_spam_pipeline.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.2
MAX_FEATURES = 10000


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

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
    """
    Extracts numeric and keyword-based features from raw text.
    """
    def fit(self, X, y=None): return self

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

print("Loading data...")
try:
    df_temp = pd.read_csv(DATA_PATH, encoding='latin-1', header=0)
    df_temp.columns = df_temp.columns.str.strip().str.lower()
    if 'v1' in df_temp.columns and 'v2' in df_temp.columns:
        df = df_temp[['v1', 'v2']]
        df.columns = ['label', 'text']
    elif 'label' in df_temp.columns and 'text' in df_temp.columns:
        df = df_temp[['label', 'text']]
    elif len(df_temp.columns) >= 2:
        df = df_temp.iloc[:, [0, 1]]
        df.columns = ['label', 'text']
    else:
        raise ValueError("CSV file structure is not supported.")

    df.dropna(subset=['text'], inplace=True)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    y = df['label'].map({'ham': 0, 'spam': 1})

    if y.isnull().any():
        valid_rows = y.notnull()
        df = df[valid_rows]
        y = y[valid_rows]

    if df.empty:
        raise ValueError("Dataset is empty after preprocessing.")

except FileNotFoundError:
    print(f"Error: File not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error during data loading: {e}")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('clean', TextCleaner()),
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 3),
                max_features=MAX_FEATURES,
                token_pattern=r'(?u)\b\w[\w-]*\b'
            ))
        ])),
        ('numeric_pipeline', Pipeline([
            ('numeric', NumericFeatures()),
            ('scaler', StandardScaler())
        ]))
    ])),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

tfidf_vectorizer = pipeline.named_steps['features'].transformer_list[0][1].named_steps['tfidf']
feature_names = tfidf_vectorizer.get_feature_names_out()
coefs = pipeline.named_steps['clf'].coef_[0]

print("TF-IDF vocabulary sample:", list(tfidf_vectorizer.vocabulary_.keys())[:20])
for word in ['cash', 'prize', 'selected', 'reward', 'won', 'claim']:
    if word in tfidf_vectorizer.vocabulary_:
        idx = tfidf_vectorizer.vocabulary_[word]
        print(f"'{word}' coefficient: {coefs[idx]:.4f}")
    else:
        print(f"'{word}' not found in vocabulary.")

top_spam = sorted(zip(coefs, feature_names), reverse=True)[:20]
print("Top 20 spam features:")
for coef, word in top_spam:
    print(f"{word}: {coef:.4f}")


os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
joblib.dump(pipeline, PIPELINE_PATH)
print(f"Pipeline saved to {PIPELINE_PATH}")



def predict_spam(texts, pipeline, threshold=THRESHOLD):
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
                "prediction": "spam" if proba >= threshold else "not spam"
            })
    return results



def evaluate_model(pipeline, X, y, threshold=THRESHOLD):
    y_probs = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    print(f"Accuracy (Threshold={threshold}): {acc:.4f}")
    print(f"Precision: {pr:.4f} | Recall: {rc:.4f} | F1: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=['ham', 'spam']))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == '__main__':
    print("Evaluating model on test set...")
    evaluate_model(pipeline, X_test, y_test, threshold=THRESHOLD)
