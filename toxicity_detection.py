import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("dataset/toxicity.csv")  

# fill missing values
df['Comment'] = df['Comment'].fillna("")

# Convert labels to binary (0 or 1) based on a threshold
label_cols = ['Toxicity', 'Severe_Toxicity', 'Identity_Attack', 'Insult', 'Profanity', 'Threat']
threshold = 0.5 #taas man accuracy ko di
y = (df[label_cols] >= threshold).astype(int)
X = df['Comment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultiOutputClassifier(LogisticRegression(solver='liblinear')))
])


pipeline.fit(X_train, y_train)


#way ko kabalo ano ang kinalainn sang joblib pero sige lang jobs 
joblib.dump(pipeline, "models/toxicity_model.joblib")

y_pred = pipeline.predict(X_test)

print("\n📊 Classification Report per Label:")
for i, col in enumerate(label_cols):
    print(f"\n--- {col} ---")
    print(classification_report(y_test[col], y_pred[:, i]))


print("\n📈 Accuracy per Label:")
for i, col in enumerate(label_cols):
    acc = accuracy_score(y_test[col], y_pred[:, i])
    print(f"{col}: {acc:.4f}")

pipeline = joblib.load("models/toxicity_model.joblib")
