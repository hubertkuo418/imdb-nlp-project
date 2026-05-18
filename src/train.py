# src/train.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("data/IMDB Dataset.csv")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# =========================
# 2. Train / Val / Test split
# =========================
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)

y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

# =========================
# 3. Tokenizer
# =========================
max_words = 20000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token="__OOV__")
tokenizer.fit_on_texts(train_df['review'])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['review']), maxlen=max_len)
X_val = pad_sequences(tokenizer.texts_to_sequences(val_df['review']), maxlen=max_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['review']), maxlen=max_len)

# save tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# =========================
# 4. RNN Model
# =========================
rnn_model = Sequential([
    Embedding(max_words, 128),
    SimpleRNN(64, kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

rnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=6,
    batch_size=128
)

rnn_model.save("models/rnn_model.keras")

# =========================
# 5. LSTM Model
# =========================
lstm_model = Sequential([
    Embedding(max_words, 128),
    LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(1e-4)),
    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=6,
    batch_size=128
)

lstm_model.save("models/lstm_model.keras")

print("Training complete & models saved.")