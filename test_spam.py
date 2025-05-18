import re
import joblib

from lib.pyModels import TextCleaner, NumericFeatures

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
                "is_spam": bool(proba >= threshold)
            })
    return results

def detect_spam(sentences: list[str]):
    pipeline = joblib.load('models/full_spam_pipeline.joblib')
    
    results = predict_spam(sentences, pipeline)
    count = len(list(filter(lambda x: x['is_spam'], results)))
    
    return results, count

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