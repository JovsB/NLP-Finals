# NLP-FINALS

## Overview

**SimpleAnalyst** is an online tool for Spam Detection, Toxicity Detection, and Sentiment Analysis. It leverages modern machine learning and natural language processing (NLP) techniques to analyze text messages for spam, toxic content, and sentiment. The project features a Python backend with Flask, scikit-learn, and NLTK, and a React-based frontend for interactive user experience.

---

## Features

- **Spam Detection:**  
  Classifies messages as spam or not spam using a logistic regression model with custom feature engineering and rule-based logic.

- **Toxicity Detection:**  
  Identifies multiple types of toxic content (e.g., toxicity, insult, threat, profanity) using a multi-label classifier.

- **Sentiment Analysis:**  
  Determines whether a message is positive, negative, or neutral, and provides a confidence score.

- **Interactive Web Interface:**  
  Built with React for real-time text analysis and results display.

---

## Technology Stack

- **Python** (Flask, scikit-learn, NLTK)
- **React** (frontend)
- **Node.js** (frontend tooling)
- **joblib** (model persistence)

---

## Project Structure
.
├── app.py
├── spam_classifier.py
├── sentiment_analysis.py
├── toxicity_detection.py
├── utils.py
├── dataset/
│   ├── spam.csv
│   └── toxicity.csv
├── models/
│   ├── full_spam_pipeline.joblib
│   ├── sentiment_model_nltk_reviews.joblib
│   └── toxicity_model.joblib
├── lib/
│   └── pyModels.py
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── .venv/
│   └── ... (virtual environment files and folders)
├── requirements.txt
└── README.md

## Authors ## 
Joven Carl Rex Biaca
Rei Ebenezer Duhina
Kyle Eron Hallares
John Paul Sapasap
Lord Patrick Raizen Togonon

## Acknowledgements ##
Sir John Christopher Mateo (Faculty Adviser)
West Visayas State University, College of Information and Technology
Open-source libraries: scikit-learn, NLTK, Flask, React