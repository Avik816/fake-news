# Fake News Detection using Machine Learning

This project aims to build a machine learning model that can classify news articles as **real** or **fake** based on their textual content. The process includes data preprocessing, feature extraction using TF-IDF, model training using Naive Bayes, and performance evaluation.

## 📌 Project Objectives

1. Load and understand the dataset  
2. Clean and preprocess the text data  
3. Extract features using TF-IDF  
4. Train a Naive Bayes classifier  
5. Evaluate model performance  
6. Save the trained model for future use

## 🗃️ Dataset

The dataset used contains labeled news data indicating whether each article is real or fake. It is assumed to be in CSV format and includes a text column and a label column.

## 🧹 Data Cleaning & Preprocessing

- **Stopwords removal**  
- **Tokenization**  
- **Regex cleaning**  
- **Lowercasing**

## 🔧 Feature Extraction

TF-IDF vectorization is used to convert text data into numerical vectors:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(preprocessed_text)
```

## 🧪 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
