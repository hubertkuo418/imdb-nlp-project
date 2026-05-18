import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# config
# =========================
MAX_LEN = 200

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬")

st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review and choose a model to predict sentiment.")

# =========================
# load tokenizer
# =========================
@st.cache_resource
def load_tokenizer():
    with open("models/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()

# =========================
# load models
# =========================
@st.cache_resource
def load_models():
    return {
        "RNN": load_model("models/rnn_model.keras"),
        "LSTM": load_model("models/lstm_model.keras")
    }

models = load_models()

# =========================
# UI
# =========================
model_choice = st.selectbox("Choose Model", ["RNN", "LSTM"])

text = st.text_area("Enter your review here:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a review.")
    else:
        model = models[model_choice]

        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        prob = model.predict(padded)[0][0]
        pred = "Positive 😊" if prob > 0.5 else "Negative 😡"

        st.subheader("Result")
        st.write(f"Prediction: {pred}")
        st.write(f"Confidence: {prob:.3f}")

        # simple bar visualization
        st.progress(float(prob) if prob > 0.5 else float(1 - prob))