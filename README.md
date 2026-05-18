# IMDB Sentiment Analysis Project (RNN vs LSTM)

## Overview
This project performs sentiment analysis on the IMDB movie review dataset using deep learning models, including:

- Simple RNN
- LSTM

The goal is to classify reviews as **positive (1)** or **negative (0)** and compare model performance.

---

## Dataset
- IMDB Large Movie Review Dataset
- 50,000 reviews
- Binary classification:
  - Positive = 1
  - Negative = 0

### Split
- Train: 80%
- Validation: 10%
- Test: 10%

---

## Models

### RNN
- Embedding layer (128 dim)
- SimpleRNN (64 units)
- Dense layers + Dropout

### LSTM
- Embedding layer (128 dim)
- LSTM (128 units)
- Dense layers + Dropout + recurrent dropout

---

## Training Config
- Loss: Binary Crossentropy
- Optimizer: Adam
- Batch size: 128
- Epochs: 6

---

## Evaluation Metrics
- Accuracy
- Precision / Recall / F1-score
- ROC-AUC
- Confusion Matrix
- WordCloud (Positive / Negative)

---

## Results Summary

| Model | Accuracy | AUC |
|------|---------|-----|
| RNN  | ~0.81   | ~0.88 |
| LSTM | ~0.87   | ~0.93 |

---

## Visualizations
- Confusion Matrix Heatmaps
- ROC Curves
- WordCloud (Positive vs Negative)

### LSTM Screenshots
Here are key screenshots from the LSTM model:

![LSTM Screenshot 1](assets/LSTM_confusion_matrix.png)
![LSTM Screenshot 2](assets/LSTM_negative_wordcloud.png)
![LSTM Screenshot 3](assets/LSTM_positive_wordcloud.png)
![LSTM Screenshot 4](assets/LSTM_roc.png)

Stored in:
```
assets/
```
- Confusion Matrix Heatmaps
- ROC Curves
- WordCloud (Positive vs Negative)

Stored in:
```
assets/
```

---

## Streamlit App
Run locally:
```bash
streamlit run app.py
```

Features:
- Input movie review
- Choose model (RNN / LSTM)
- Real-time prediction

---

## Project Structure
```
imdb-nlp-project/
├── data/
├── models/
├── src/
│   ├── train.py
│   ├── evaluate.py
├── assets/
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Train model
```bash
python src/train.py
```

### 2. Evaluate model
```bash
python src/evaluate.py
```

### 3. Run web app
```bash
streamlit run app.py
```

---

## Key Takeaways
- LSTM significantly outperforms RNN
- RNN suffers from long-term dependency issues
- WordCloud helps interpret sentiment patterns

---

## Author
HubertKuo

