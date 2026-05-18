# src/evaluate.py

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)

from wordcloud import WordCloud

# =========================
# config
# =========================
MAX_LEN = 200
ASSETS_DIR = "assets"

os.makedirs(ASSETS_DIR, exist_ok=True)

# =========================
# load data (test only for evaluation)
# =========================
df = pd.read_csv("data/IMDB Dataset.csv")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

from sklearn.model_selection import train_test_split
_, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
_, test_df = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

X_test_text = test_df["review"].values
y_test = test_df["label"].values

# =========================
# load tokenizer
# =========================
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN)

# =========================
# models
# =========================
models = {
    "RNN": load_model("models/rnn_model.keras"),
    "LSTM": load_model("models/lstm_model.keras")
}

# =========================
# helper: plot confusion matrix
# =========================
def plot_cm(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"])
    disp.plot(cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"{ASSETS_DIR}/{name}_confusion_matrix.png")
    plt.close()

# =========================
# helper: ROC curve
# =========================
def plot_roc(y_true, y_score, name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.savefig(f"{ASSETS_DIR}/{name}_roc.png")
    plt.close()

    return roc_auc

# =========================
# helper: wordcloud
# =========================
def plot_wordcloud(texts, name):
    text = " ".join(texts)

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{name} WordCloud")

    plt.savefig(f"{ASSETS_DIR}/{name}_wordcloud.png")
    plt.close()

# =========================
# evaluation loop
# =========================
for name, model in models.items():

    print(f"\n===== Evaluating {name} =====")

    # prediction probability
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)


    # classification report
    print(classification_report(y_test, y_pred))

    # confusion matrix
    plot_cm(y_test, y_pred, name)

    # ROC
    auc_score = plot_roc(y_test, y_prob, name)
    print(f"{name} AUC: {auc_score:.3f}")
    
    # accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    # WordCloud (positive + negative split)
    pos_text = X_test_text[y_test == 1]
    neg_text = X_test_text[y_test == 0]

    plot_wordcloud(pos_text, f"{name}_positive")
    plot_wordcloud(neg_text, f"{name}_negative")

print("\nEvaluation complete. All outputs saved to assets/")