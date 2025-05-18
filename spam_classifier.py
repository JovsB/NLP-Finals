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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

 
DATA_PATH = 'dataset/spam.csv'
PIPELINE_PATH = 'models/full_spam_pipeline.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.6
MAX_FEATURES = 5000

 
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
        return re.sub(r'\s+', ' ', text).strip()

class NumericFeatures(BaseEstimator, TransformerMixin):
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

        return np.hstack([exclaim_count, has_free, has_url, digit_count, uppercase_ratio])

 
print("Loading data...")
try:
    df_temp = pd.read_csv(DATA_PATH, encoding='latin-1')

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
    df['label'] = df['label'].astype(str)
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

#  Train-Test Split---
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# pipeline
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            ('clean', TextCleaner()),
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=MAX_FEATURES,
                token_pattern=r'(?u)\b\w[\w-]*\b'
            ))
        ])),
        ('numeric_pipeline', Pipeline([
            ('numeric', NumericFeatures())
        ]))
    ])),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ))
])

 
print("Training model...")
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, PIPELINE_PATH)
print(f"Pipeline saved to {PIPELINE_PATH}")

 
def predict_spam(text: str, threshold: float = THRESHOLD) -> bool:
    proba = pipeline.predict_proba([text])[0, 1]
    return proba >= threshold

 
if __name__ == '__main__':
    print("Evaluating model on test set...")
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= THRESHOLD).astype(int)

    print(f"Accuracy (Threshold={THRESHOLD}): {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
