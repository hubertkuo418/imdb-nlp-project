# src/predict.py

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# config
# =========================
max_len = 200

# =========================
# load tokenizer
# =========================
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# load model
# =========================
model_name = input("Choose model (rnn / lstm): ").lower()

if model_name == "rnn":
    model = load_model("models/rnn_model.keras")
elif model_name == "lstm":
    model = load_model("models/lstm_model.keras")
else:
    raise ValueError("model must be rnn or lstm")

# =========================
# predict loop
# =========================
while True:
    text = input("\nEnter review (or type quit): ")

    if text == "quit":
        break

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    pred = model.predict(padded)[0][0]

    label = "positive " if pred > 0.5 else "negative "

    print(f"Prediction: {label} (score={pred:.3f})")