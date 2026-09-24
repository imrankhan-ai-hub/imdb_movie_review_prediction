import streamlit as st
import joblib


# Load trained model

MODEL_PATH = "models/imdb_sentiment_model.pkl"

model = joblib.load(MODEL_PATH)


# Page configuration

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# Header

st.title("🎬 IMDB Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review below and the machine learning "
    "model will predict whether the sentiment is Positive or Negative."
)


# Review input

review = st.text_area(
    "Enter your movie review:",
    height=200,
    placeholder="Example: This movie was absolutely amazing..."
)


# Prediction


if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():

        st.warning(
            "Please enter a movie review before analyzing."
        )

    else:

        prediction = model.predict([review])[0]

     
        if prediction == 1:

            st.success("😊 Positive Review")

            

        else:

            st.error("😞 Negative Review")

            


# Information section


st.divider()

st.subheader("About the Model")

st.write(
    """
    This application uses a machine-learning sentiment
    classification pipeline trained on the IMDB 50K Movie
    Reviews dataset.

    Text is converted into numerical TF-IDF features and
    classified using Logistic Regression.
    """
)