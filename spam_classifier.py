import pandas as pd
import re
import numpy as np
import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# config
DATA_PATH = 'models/spam.csv'
PIPELINE_PATH = 'models/full_spam_pipeline.joblib'
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.6     # adjust for tradeoff
MAX_FEATURES = 5000  # limit TF-IDF features to top 5k

# Custom Text Cleaner Transformer 
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(self.clean_text)
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

# Custom Numeric Features Transformer 
class NumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        exclaim_count = X.str.count('!').values.reshape(-1, 1)
        has_free = X.str.contains(r'\bfree\b').astype(int).values.reshape(-1, 1)
        return np.hstack([exclaim_count, has_free])

#  load prepare data
print("Loading data...")
df = pd.read_csv(DATA_PATH, encoding='latin-1', usecols=[0, 1], names=['label', 'text'], header=0)
y = df['label'].map({'ham': 0, 'spam': 1})

# Split raw text and labels 
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

#  Build end-to-end sklearn Pipeline with FeatureUnion
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('clean', TextCleaner()),
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=MAX_FEATURES
            ))
        ])),
        ('numeric', Pipeline([ 
            ('num', NumericFeatures())
        ]))
    ])),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ))
])

# train pipline
print("Training pipeline...")
pipeline.fit(X_train, y_train)

# save pipeline
joblib.dump(pipeline, PIPELINE_PATH)
print(f"Pipeline saved to {PIPELINE_PATH}")

#  Define prediction function
pipeline = joblib.load(PIPELINE_PATH)

def predict_spam(text: str, threshold: float = THRESHOLD) -> bool:
    proba = pipeline.predict_proba([text])[0, 1]
    return proba >= threshold

# evaluate test set
if __name__ == '__main__':
    print("Evaluating on test set...")
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= THRESHOLD).astype(int)

    print(f"Accuracy ({THRESHOLD}): {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
