#  IMDB Movie Review Sentiment Analysis

A complete Natural Language Processing (NLP) project that classifies IMDB movie reviews as **Positive** or **Negative** using **TF-IDF Vectorization** and **Logistic Regression**.

The project includes data preprocessing, exploratory data analysis, text cleaning, feature extraction, model training, evaluation, model serialization, and a Streamlit-based web application.



##  Project Overview

Sentiment analysis is a Natural Language Processing task used to determine the emotional polarity of text.

In this project, a machine learning model is trained to analyze movie reviews and predict whether a review expresses:

- Positive sentiment
- Negative sentiment

The project uses the **IMDB Dataset of 50K Movie Reviews**.

---

##  Dataset

**Dataset:** IMDB Dataset of 50K Movie Reviews

The dataset contains:

- **50,000 movie reviews**
- Review text
- Sentiment label
- Two classes:
  - `positive`
  - `negative`

Each review is classified into one of the two sentiment categories.

---

##  Project Workflow

```text
IMDB Dataset
     ↓
Data Loading
     ↓
Data Inspection
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Train/Test Split
     ↓
TF-IDF Feature Extraction
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Model save
     ↓
Streamlit Web Application
```

---

##  Data Preprocessing

The text preprocessing workflow includes:

- Checking missing values
- Checking duplicate reviews
- Removing HTML-related content
- Cleaning unnecessary URLs and characters
- Normalizing whitespace
- Converting sentiment labels into numerical form
- Examining review length and text distribution

The cleaned review text is then passed to the TF-IDF vectorizer.

---

## Feature Engineering

### TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** is used to convert movie reviews into numerical features that machine learning algorithms can understand.

The final vectorizer uses:

```python
TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

The model considers both:

- Unigrams — individual words
- Bigrams — pairs of words

For example:

```text
"excellent movie"
```

can generate features such as:

```text
excellent
movie
excellent movie
```

---

##  Machine Learning Model

### Logistic Regression

Logistic Regression is used as the final classification model.

It was selected because it is:

- Efficient for high-dimensional text data
- Computationally efficient
- Well suited for binary classification
- Easy to deploy
- Compatible with sparse TF-IDF features

Multiple baseline models were experimented with during development. Their performance was broadly similar on the test data, so Logistic Regression with TF-IDF was selected as the final deployment model.

No hyperparameter tuning or cross-validation was used because the baseline models already produced satisfactory and broadly similar results for this project.

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

The confusion matrix helps identify:

```text
True Positive
True Negative
False Positive
False Negative
```

Actual evaluation values are generated directly from the test set in the notebook.

---

## Model Saving

The complete TF-IDF + Logistic Regression pipeline is saved using `joblib`.

```text
models/
└── imdb_sentiment_model.pkl
```

Saving the complete pipeline allows the application to directly process raw review text without manually recreating the TF-IDF transformation.

---

## 🖥️ Streamlit Web Application

The project includes a simple interactive web application built with Streamlit.

Users can:

1. Enter a movie review
2. Click **Analyze Sentiment**
3. Receive the predicted sentiment


Example:

```text
Input:
"This movie was absolutely fantastic. The acting was brilliant."

Output:
 Positive Review
```

---

## 📁 Project Structure

```text
imdb_sentiment_analysis/
│
├── models/
│   └── imdb_sentiment_model.pkl
│
├── app.py
│
├── notebook.ipynb
│
├── README.md
│
└── env/
```

> The `env/` virtual environment should not be uploaded to GitHub. It should be listed in `.gitignore`.

