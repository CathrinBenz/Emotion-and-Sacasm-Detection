# Emotion-and-Sacasm-Detection
An NLP-based machine learning project that detects emotions and sarcasm in text, enabling emotionally intelligent AI communication.
# 💬 EMOTION AND SARCASM DETECTION SYSTEM
# 🎯 Objective
The primary objective of this project is to develop a machine learning-based system capable of detecting emotions and sarcasm in textual data. The system combines Natural Language Processing (NLP) and Machine Learning (ML) techniques to accurately identify emotional states such as joy, anger, sadness, fear, disgust, shame, guilt and recognize sarcastic expressions where literal meanings contradict intended sentiments
This includes:

Preprocessing raw text to remove noise and normalize input.

Extracting features using n-grams, TF-IDF, and word embeddings.

Training and evaluating ML models (LinearSVC & Logistic Regression).

Combining emotion and sarcasm detection to achieve more reliable sentiment interpretation.

Testing and validating model accuracy and efficiency.

The ultimate goal is to enable emotionally intelligent systems that understand human feelings and sarcasm for real-world applications like chatbots, customer feedback analysis, and social media monitoring.
# 📑 Table of Contents

Introduction & Objective

Literature Review

Proposed System

System Design

Software Description

Program Design

Testing

Conclusion

References

Appendix (Source Code, Sample Outputs, Screenshots)

# 📘 Project Overview

In today’s digital era, people frequently share emotions and opinions online through platforms like Twitter, Facebook, and chat applications. Understanding these emotions accurately is crucial for customer feedback systems, mental health monitoring, and social media analytics.

Traditional sentiment analysis often fails to capture sarcasm, where the literal meaning differs from the intended emotion (e.g., “Oh great, my phone died again!”).

This project overcomes these challenges by developing an Emotion and Sarcasm Detection System using Python. It employs Natural Language Processing (NLP) and Machine Learning models to classify text into emotional categories and detect sarcastic remarks. The system learns linguistic and contextual cues from labeled datasets to predict both emotion and sarcasm effectively.

# ⚙️ Features

✅ Dual-Module Design: Emotion Detection + Sarcasm Detection ✅ Uses NLP preprocessing (tokenization, stopword removal, normalization) ✅ Employs LinearSVC and Logistic Regression for classification ✅ Handles sarcasm through rule-based and ML-based detection ✅ Supports real-time user input for prediction ✅ Outputs both emotion category and sarcasm status ✅ Achieved ~85% accuracy in emotion detection and ~80% accuracy in sarcasm detection

🧩 Tools & Technologies Used Category Tools / Technologies Programming Language Python 3.x Libraries scikit-learn, pandas, numpy, re, joblib, collections Algorithms Linear Support Vector Classifier (SVC), Logistic Regression Feature Extraction n-grams, TF-IDF IDEs Google Colab / VS Code / PyCharm Dataset Emotion & Sarcasm labeled datasets Output Interface Command-line console 💾 Dataset Description

Emotion Dataset: Contains labeled sentences representing multiple emotional states – joy, fear, anger, sadness, disgust, shame, guilt.

Sarcasm Dataset: Labeled comments marked as “Sarcastic” or “Not Sarcastic.”

Data is cleaned, tokenized, and vectorized using n-gram and TF-IDF features for model training.

# 🧠 Proposed System

The system processes raw text, removes noise, and transforms it into numerical vectors. It consists of two main components:

1️⃣ Emotion Detection Module

Uses n-gram features and Linear Support Vector Classifier (SVC).

Classifies sentences into one or more emotional categories.

2️⃣ Sarcasm Detection Module

Uses TF-IDF features with Logistic Regression.

Incorporates rule-based heuristics (e.g., phrases like “yeah right”, “oh great”).

Detects sarcasm even in indirect or implicit cases.

# 🧩 System Design 🔹 Data Flow

User Input → User enters a sentence.

Preprocessing → Cleans text (removes punctuation, symbols, expands contractions).

Feature Extraction → Converts text into vectorized format (n-grams / TF-IDF).

Model Prediction →

Emotion model predicts the emotional state.

Sarcasm model predicts sarcastic intent.

Output Display → System displays both detected emotion and sarcasm status.

# 🧱 Architecture

Input Layer: Text sentence

Processing Layer: NLP preprocessing & feature extraction

Model Layer: Trained SVC and Logistic Regression models

Output Layer: Emotion + Sarcasm prediction

# 💻 Program Design 🧩 Emotion Detection Module

Model: LinearSVC (Support Vector Machine)

Feature Extraction: n-grams + DictVectorizer

Dataset Split: 70% training / 30% testing

Output Example:

Input: “Everything feels perfect today”
Output: Emotion → Joy

# 🧩 Sarcasm Detection Module

Model: Logistic Regression

Feature Extraction: TF-IDF (1–2 grams)

Dataset Split: 80% training / 20% testing

Output Example:

Input: “Oh wonderful, it’s raining again on my day off.”
Output: Sarcastic

🧪 Model Training & Evaluation Module Algorithm Feature Extraction Accuracy Emotion Detection Linear SVC n-gram + CountVectorizer ~85% Sarcasm Detection Logistic Regression TF-IDF (1–2 grams) ~80%

Evaluation Metrics:

Precision: 0.84 (Emotion), 0.79 (Sarcasm)

Recall: 0.83 (Emotion), 0.80 (Sarcasm)

F1-Score: 0.84 (Emotion), 0.79 (Sarcasm)

# 🔬 Testing ✅ Test Plan

Validate emotion classification accuracy.

Verify sarcasm detection functionality.

Check for handling of invalid or empty inputs.

Confirm integration between both models.

✅ Test Results Input Detected Emotion Sarcasm “Everything feels perfect today.” Joy Not Sarcastic “Oh great, another meeting at 7 a.m.!” Disgust Sarcastic “I can’t even look at myself after what I did.” Shame Not Sarcastic 📊 Performance Summary

Emotion Detection Model

Training Accuracy: ~90%

Testing Accuracy: ~85%

Sarcasm Detection Model

Training Accuracy: ~87%

Testing Accuracy: ~80%

Both models show strong generalization and minimal overfitting. TF-IDF features captured phrase-level sarcasm effectively, while n-grams captured emotional tone.

# 💡 Applications

✅ Customer Feedback Analysis ✅ Social Media Sentiment Tracking ✅ Chatbots & Virtual Assistants ✅ Mental Health Text Monitoring ✅ Emotionally Intelligent AI Systems

# 🚀 Insights & Future Scope

Future improvements include using deep learning (LSTM, BERT) for better contextual understanding.

Multilingual support can enhance usability across languages.

Integration into real-time chatbots or social media dashboards could provide live emotion & sarcasm analysis.

# ✅ Conclusion

This project successfully demonstrates how machine learning and NLP can be combined to detect both emotions and sarcasm in textual data. By integrating these two aspects, the system achieves a deeper understanding of human communication, enabling applications in marketing, social analytics, and human-computer interaction.

The dual-model framework provides a reliable approach for emotion classification and sarcasm detection, paving the way for emotionally aware AI systems capable of understanding nuanced human language.

# 📚 References

Investigations in Computational Sarcasm — Aditya Joshi et al., 2018
Emotion Detection in NLP — Federica Cavicchio, 2024

Sentiment Analysis: Mining Opinions and Emotions — Bing Liu, 2015

Deep Learning Approaches for Sentiment Analysis — Basant Agarwal et al., 2020

Computational Intelligence Methods for Sentiment Analysis — D. Jude Hemanth, 2024

# Sample Code
# SARCASM CODE

import pandas as pd

import re

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import joblib

# Step 1: Load dataset

df = pd.read_csv("/content/sarcasm_dataset_500.csv")

print("Dataset Loaded Successfully!")


# Step 2: Split data

X = df['text']

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: TF-IDF Vectorization

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

# Step 4: Train Logistic Regression Model

model = LogisticRegression(max_iter=1000, C=2)

model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save Model

joblib.dump(model, "sarcasm_model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

# Step 7: Predict Function

def predict_sarcasm(comment):

    sarcasm_keywords = [
        "yeah right", "oh great", "of course", "just what i needed",
        "sure thing", "as if", "can't wait", "amazing not", 
        "totally helpful", "wonderful news", "nice job", "brilliant idea"
    ]
    comment_clean = comment.lower().strip()
    if any(phrase in comment_clean for phrase in sarcasm_keywords):
        return "Sarcastic"
    comment_vec = vectorizer.transform([comment])
    prediction = model.predict(comment_vec)[0]
    return "Sarcastic" if prediction == 1 else "Not Sarcastic"


# EMOTION CODE 

import re

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

from sklearn.feature_extraction import DictVectorizer

import joblib

# Step 1: Read dataset

def read_data(file):

    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

# Step 2: Feature extraction using n-grams

def ngram(token, n):

    return [' '.join(token[i-n+1:i+1]) for i in range(n-1, len(token))]

def create_feature(text, nrange=(1, 3)):

    text = text.lower()
    text = re.sub(r"i['’]m", "i am", text)
    text = re.sub(r"can['’]t", "cannot", text)
    text = re.sub(r"n['’]t", " not", text)
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    text_punc = re.sub('[a-z0-9]', ' ', text)
    features = []
    for n in range(nrange[0], nrange[1] + 1):
        features += ngram(text_alphanum.split(), n)
    features += ngram(text_punc.split(), 1)
    return Counter(features)

# Step 3: Train Model

file = "/content/emotions_dataset_realistic_ordered.txt"

data = read_data(file)

emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

X_all, y_all = [], []

for label, text in data:

    y_all.append(label)
    X_all.append(create_feature(text, nrange=(1, 3)))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=123)

vectorizer = DictVectorizer(sparse=True)

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

clf = LinearSVC(random_state=123, class_weight='balanced')

clf.fit(X_train, y_train)

print(f"Training Accuracy: {accuracy_score(y_train, clf.predict(X_train)):.2f}")

print(f"Test Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.2f}")

# Save Model

joblib.dump(clf, "emotion_model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

# Prediction

def predict_emotion(sentence):

    features = create_feature(sentence, nrange=(1, 3))
    vec = vectorizer.transform([features])
    return clf.predict(vec)[0]

