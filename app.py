# app.py -- à lancer avec : streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

VOCAB_SIZE = 10000
MAX_LEN = 200


# === Cache des modèles ===

@st.cache_resource
def load_imdb_model():
    return keras.models.load_model('imdb_sentiment.keras')

@st.cache_resource
def load_airline_lstm():
    return keras.models.load_model('airline_lstm.keras')

@st.cache_resource
def load_airline_gru():
    return keras.models.load_model('airline_gru.keras')

@st.cache_resource
def load_word_index():
    return imdb.get_word_index()


def preprocess_text(text, word_index, max_len=MAX_LEN, vocab_size=VOCAB_SIZE):
    tokens = text.lower().split()
    sequence = []
    for word in tokens:
        idx = word_index.get(word, 0) + 3
        if idx < vocab_size:
            sequence.append(idx)
        else:
            sequence.append(2)
    padded = pad_sequences([sequence], maxlen=max_len, padding='pre', truncating='pre')
    return padded


# === Interface ===

st.title("🧠 Deep Learning Dashboard")

tab1, tab2 = st.tabs(["🎬 Sentiment IMDB", "✈️ Airline LSTM vs GRU"])

# =============================================
# TAB 1 : Sentiment IMDB (phase 7)
# =============================================
with tab1:
    st.header("Analyse de Sentiment IMDB")
    st.caption("Note : les textes de plus de 200 mots sont tronqués.")

    model_imdb = load_imdb_model()
    word_index = load_word_index()

    user_input = st.text_area(
        "Votre review :",
        placeholder="Ex: This movie was absolutely fantastic...",
        height=150
    )

    if st.button("🔍 Analyser le sentiment"):
        if not user_input.strip():
            st.warning("⚠️ Veuillez entrer un texte.")
        else:
            padded = preprocess_text(user_input, word_index)
            score = model_imdb.predict(padded, verbose=0)[0][0]
            label = "Positif" if score > 0.5 else "Négatif"
            confidence = score if score > 0.5 else 1 - score

            if label == "Positif":
                st.success(f"✅ Sentiment : **{label}** (confiance : {confidence:.1%})")
            else:
                st.error(f"❌ Sentiment : **{label}** (confiance : {confidence:.1%})")

            st.progress(float(score))
            st.caption(f"Score brut : {score:.4f}")

# =============================================
# TAB 2 : Airline Passengers LSTM vs GRU
# =============================================
with tab2:
    st.header("Comparaison LSTM vs GRU — Airline Passengers")

    lstm_model = load_airline_lstm()
    gru_model = load_airline_gru()

    # Charger les données Airline Passengers
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url)
    data = df['Passengers'].values.astype('float32').reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    WINDOW_SIZE = 12
    split = int(len(data_scaled) * 0.67)

    # Créer les séquences test
    def create_dataset(dataset, window_size=12):
        X, y = [], []
        for i in range(len(dataset) - window_size):
            X.append(dataset[i:i + window_size, 0])
            y.append(dataset[i + window_size, 0])
        return np.array(X).reshape(-1, window_size, 1), np.array(y)

    X_all, y_all = create_dataset(data_scaled, WINDOW_SIZE)

    # Prédictions des deux modèles
    lstm_pred = scaler.inverse_transform(lstm_model.predict(X_all, verbose=0))
    gru_pred = scaler.inverse_transform(gru_model.predict(X_all, verbose=0))
    y_real = scaler.inverse_transform(y_all.reshape(-1, 1))

    # RMSE test
    test_start = split - WINDOW_SIZE
    rmse_lstm = np.sqrt(np.mean((y_real[test_start:] - lstm_pred[test_start:]) ** 2))
    rmse_gru = np.sqrt(np.mean((y_real[test_start:] - gru_pred[test_start:]) ** 2))

    # Métriques côte à côte
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LSTM — RMSE test", f"{rmse_lstm:.1f} passagers")
        st.metric("LSTM — Paramètres", f"{lstm_model.count_params():,}")
    with col2:
        st.metric("GRU — RMSE test", f"{rmse_gru:.1f} passagers")
        st.metric("GRU — Paramètres", f"{gru_model.count_params():,}")

    # Graphique comparatif
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(y_real)), y_real, color='gray', alpha=0.6, label='Ground Truth', linewidth=2)
    ax.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(lstm_pred)), lstm_pred, color='blue', alpha=0.7, label=f'LSTM (RMSE={rmse_lstm:.1f})', linestyle='--')
    ax.plot(range(WINDOW_SIZE, WINDOW_SIZE + len(gru_pred)), gru_pred, color='red', alpha=0.7, label=f'GRU (RMSE={rmse_gru:.1f})', linestyle='--')
    ax.axvline(x=split, color='black', linestyle=':', alpha=0.5, label='Split train/test')
    ax.set_title("LSTM vs GRU — Prédictions Airline Passengers")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Passagers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Quel modèle gagne ?
    winner = "LSTM" if rmse_lstm < rmse_gru else "GRU"
    diff = abs(rmse_lstm - rmse_gru)
    st.info(f"🏆 **{winner}** gagne avec {diff:.1f} passagers de RMSE en moins")