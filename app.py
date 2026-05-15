# app.py -- à lancer avec : streamlit run app.py

import streamlit as st
import numpy as np
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
MAX_LEN = 200
MODEL_PATH = 'imdb_sentiment.keras'


@st.cache_resource
def load_model():
    """Charge le modèle une seule fois (mis en cache)."""
    return keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_word_index():
    """Charge le dictionnaire mot -> index IMDB."""
    word_index = imdb.get_word_index()
    return word_index


def preprocess_text(text, word_index, max_len=MAX_LEN, vocab_size=VOCAB_SIZE):
    """Tokenize un texte brut et retourne une séquence paddée."""
    tokens = text.lower().split()
    # Les indices IMDB sont décalés de +3
    sequence = []
    for word in tokens:
        idx = word_index.get(word, 0) + 3
        if idx < vocab_size:
            sequence.append(idx)
        else:
            sequence.append(2)  # OOV
    padded = pad_sequences([sequence], maxlen=max_len, padding='pre', truncating='pre')
    return padded


# === Interface Streamlit ===

st.title("🎬 Analyse de Sentiment IMDB")
st.write("Entrez une review de film en anglais et le modèle prédit si elle est positive ou négative.")
st.caption("Note : les textes de plus de 200 mots sont tronqués.")

# Charger modèle et vocab
model = load_model()
word_index = load_word_index()

# Zone de saisie
user_input = st.text_area(
    "Votre review :",
    placeholder="Ex: This movie was absolutely fantastic, the acting was brilliant...",
    height=150
)

# Bouton d'analyse
if st.button("🔍 Analyser"):
    if not user_input.strip():
        st.warning("⚠️ Veuillez entrer un texte avant d'analyser.")
    else:
        # Preprocessing + prédiction
        padded = preprocess_text(user_input, word_index)
        score = model.predict(padded, verbose=0)[0][0]

        label = "Positif" if score > 0.5 else "Négatif"
        confidence = score if score > 0.5 else 1 - score

        # Affichage du résultat
        if label == "Positif":
            st.success(f"✅ Sentiment : **{label}** (confiance : {confidence:.1%})")
        else:
            st.error(f"❌ Sentiment : **{label}** (confiance : {confidence:.1%})")

        # Barre de score
        st.progress(float(score))
        st.caption(f"Score brut : {score:.4f} (seuil : 0.5)")