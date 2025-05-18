import nltk
import random
from nltk.corpus import movie_reviews, twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os
import numpy as np

# --- Model and Data Configuration ---
MODEL_DIR = "models"
MODEL_FILENAME = "sentiment_model_nltk_reviews.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# --- NLTK Resource Download ---
def download_nltk_resources():
    """Downloads necessary NLTK resources if not already present."""
    try:
        nltk.data.find("corpora/movie_reviews")
    except nltk.downloader.DownloadError:
        print("Downloading NLTK movie_reviews corpus...")
        nltk.download("movie_reviews")
    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.DownloadError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/twitter_samples.zip")
    except nltk.downloader.DownloadError:
        print("Downloading NLTK twitter_samples corpus...")
        nltk.download("twitter_samples")

# --- Data Preparation ---
def load_nltk_movie_reviews():
    """Loads and prepares the NLTK movie reviews dataset."""
    print("Loading NLTK movie_reviews data...")
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((" ".join(movie_reviews.words(fileid)), category))
    random.shuffle(documents)
    texts = [doc for doc, category in documents]
    labels = [category for doc, category in documents]
    print(f"Loaded {len(texts)} documents from NLTK movie_reviews.")
    return texts, labels

def load_nltk_twitter_samples():
    """Loads and prepares the NLTK twitter_samples dataset."""
    print("Loading NLTK twitter_samples data...")
    positive_tweets = twitter_samples.strings("positive_tweets.json")
    negative_tweets = twitter_samples.strings("negative_tweets.json")

    texts = positive_tweets + negative_tweets
    labels = ["pos"] * len(positive_tweets) + ["neg"] * len(negative_tweets)

    # Shuffle the combined tweets and labels together
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined) if combined else ([], [])

    print(f"Loaded {len(texts)} documents from NLTK twitter_samples.")
    return list(texts), list(labels)

def load_sample_custom_data():
    """Loads a small sample custom dataset. 
    In a real scenario, this function would load data from a file (CSV, JSON, etc.) 
    or another source.
    Labels must be 'pos' or 'neg' for the current model.
    """
    print("Loading sample custom data...")
    custom_data = [
        ("This is a truly wonderful experience, I am so happy!", "pos"),
        ("I'm very pleased with the outcome.", "pos"),
        ("This is fantastic news!", "pos"),
        ("What a terrible situation, I'm really upset.", "neg"),
        ("I am extremely disappointed with this product.", "neg"),
        ("This is just awful.", "neg"),
        ("The service was exceptional, highly recommended.", "pos"),
        ("A complete waste of time and money.", "neg"),
        ("I couldn't be happier with the results.", "pos"),
        ("The quality is shockingly bad.", "neg"),
        ("It's an okay product, nothing special.", "neg"), 
        ("The book was quite engaging for the most part.", "pos") 
    ]
    random.shuffle(custom_data)
    texts = [text for text, label in custom_data]
    labels = [label for text, label in custom_data]
    print(f"Loaded {len(texts)} documents from sample custom data.")
    return texts, labels

def load_and_prepare_all_data():
    """Loads and prepares data from all configured sources."""
    print("Loading and preparing all data sources for training...")
    
    all_texts = []
    all_labels = []

    # Load from NLTK movie reviews
    nltk_texts_reviews, nltk_labels_reviews = load_nltk_movie_reviews()
    all_texts.extend(nltk_texts_reviews)
    all_labels.extend(nltk_labels_reviews)

    # Load from NLTK twitter samples
    nltk_texts_tweets, nltk_labels_tweets = load_nltk_twitter_samples()
    all_texts.extend(nltk_texts_tweets)
    all_labels.extend(nltk_labels_tweets)

    # Load from sample custom data (OPTIONAL)
    # If you have your own data, you can enable this section.
    # Make sure load_sample_custom_data() is implemented to load your data.
    # print("Attempting to load custom data (if enabled)...")
    # try:
    #     custom_texts, custom_labels = load_sample_custom_data()
    #     if custom_texts and custom_labels:
    #         all_texts.extend(custom_texts)
    #         all_labels.extend(custom_labels)
    #         print(f"Successfully added {len(custom_texts)} custom documents.")
    #     else:
    #         print("Custom data loading skipped or returned empty.")
    # except Exception as e:
    #     print(f"Error loading custom data: {e}. Skipping custom data.")
    # --- End of OPTIONAL custom data section ---
    
    # Shuffle the combined dataset
    if all_texts and all_labels:
        combined = list(zip(all_texts, all_labels))
        random.shuffle(combined)
        all_texts, all_labels = zip(*combined)
        print(f"Total documents for training after combining and shuffling: {len(all_texts)}")
        return list(all_texts), list(all_labels)
    else:
        print("No data loaded from any source.")
        return [], []

# --- Model Training ---
def train_sentiment_model(X_texts, y_labels):
    """Trains the sentiment analysis model and saves it."""
    print("Training sentiment analysis model...")
    # Create a pipeline: TF-IDF Vectorizer -> Multinomial Naive Bayes Classifier
    model_pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", min_df=5, max_df=0.7, ngram_range=(1, 2)),
        MultinomialNB()
    )
    model_pipeline.fit(X_texts, y_labels)
    
    # Save the trained model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return model_pipeline

# --- Sentiment Analysis ---
class SentimentAnalyzer:
    """
    A class to perform sentiment analysis using a trained model.
    Classifies text into Positive, Negative, or Neutral and provides confidence scores.
    """
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()
        self.UNCERTAIN_THRESHOLD = 0.65 
        self.NEUTRAL_MARGIN = 0.2     

    def _load_model(self):
        """Loads the trained model from disk. Trains a new one if not found."""
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}...")
            return joblib.load(self.model_path)
        else:
            print("No pre-trained model found. Training a new one...")
            download_nltk_resources()
            X_texts, y_labels = load_and_prepare_all_data()
            if not X_texts:
                raise ValueError("No training data loaded. Cannot train model.")
            return train_sentiment_model(X_texts, y_labels)

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text and returns the label and scores.

        Args:
            text: The input string to analyze.

        Returns:
            A dictionary with keys: "label" (str), "positive_score" (float),
            "negative_score" (float).
            Example: {"label": "Positive", "positive_score": 0.9, "negative_score": 0.1}
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        
        label = "Neutral" # Default to Neutral
        pos_proba = 0.0
        neg_proba = 0.0

        if not text.strip(): # Empty or whitespace-only strings
            return {"label": "Neutral", "positive_score": 0.0, "negative_score": 0.0}

        prediction_proba = self.model.predict_proba([text])[0]
        
        classes = list(self.model.classes_)
        try:
            neg_idx = classes.index("neg")
            neg_proba = prediction_proba[neg_idx]
        except ValueError:
            print("Warning: 'neg' class not found in model. Scores might be inaccurate.")

        try:
            pos_idx = classes.index("pos")
            pos_proba = prediction_proba[pos_idx]
        except ValueError:
            print("Warning: 'pos' class not found in model. Scores might be inaccurate.")
            
        max_proba = max(pos_proba, neg_proba)

        if max_proba < self.UNCERTAIN_THRESHOLD:
            label = "Neutral"
        elif abs(pos_proba - neg_proba) < self.NEUTRAL_MARGIN:
            label = "Neutral"
        elif pos_proba > neg_proba:
            label = "Positive"
        else:
            label = "Negative"
        
        return {
            "label": label,
            "positive_score": round(pos_proba, 4), # Rounded for readability
            "negative_score": round(neg_proba, 4)
        }

    def retrain_model(self):
        """Forces retraining of the model."""
        print("Retraining model...")
        download_nltk_resources()
        X_texts, y_labels = load_and_prepare_all_data()
        self.model = train_sentiment_model(X_texts, y_labels)
        print("Model retraining complete.")


# --- Main Execution ---
if __name__ == "__main__":
    download_nltk_resources()
    analyzer = SentimentAnalyzer()

    print("\n--- Sentiment Analysis CLI ---")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'retrain' to force model retraining.")

    while True:
        user_input = input("\nEnter text to analyze: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting sentiment analyzer.")
            break
        if user_input.lower() == "retrain":
            analyzer.retrain_model()
            continue
        
        if not user_input.strip():
            print("Sentiment: Neutral (empty input), Positive Score: 0.0, Negative Score: 0.0")
            continue

        try:
            result = analyzer.analyze_sentiment(user_input)
            print(f"Sentiment: {result['label']}")
            print(f"  Positive Score: {result['positive_score']:.2%}") # Display as percentage
            print(f"  Negative Score: {result['negative_score']:.2%}")
        except Exception as e:
            print(f"Error during analysis: {e}")
            print("This might happen if the model classes are unexpected. Consider retraining.")

    # Example texts for testing (can be uncommented)
    # print("\n--- Example Texts (with scores) ---")
    # texts_to_test = [
    #     "This movie was absolutely fantastic, a true masterpiece!",
    #     "I hated every moment of that film. It was a disaster.",
    #     "The acting was okay, but the plot was a bit predictable.",
    #     "It's a film.",
    #     "I'm not really sure what to think about this movie yet.",
    #     "The product is average, neither good nor bad.",
    #     "I am incredibly happy with this purchase!",
    #     "This is the worst decision I have ever made.",
    #     "This is one of the best coffee I've ever tasted!"
    # ]
    # for t in texts_to_test:
    #     analysis_result = analyzer.analyze_sentiment(t)
    #     print(f"Text: '{t}' -> Label: {analysis_result['label']}, Pos: {analysis_result['positive_score']:.0%}, Neg: {analysis_result['negative_score']:.0%}")
