import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import emoji
import matplotlib.pyplot as plt


def limpar_texto(texto):
    texto = str(texto)
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"#", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def contar_emojis(texto):
    return sum(1 for char in str(texto) if char in emoji.EMOJI_DATA)


def extrair_features(texto_original):
    texto_original = str(texto_original)
    texto_limpo = limpar_texto(texto_original)

    palavras = texto_limpo.split()
    total_palavras = len(palavras)

    if total_palavras == 0:
        total_palavras = 1

    tamanhos_palavras = [len(p) for p in palavras]

    avg_word_length = np.mean(tamanhos_palavras) if tamanhos_palavras else 0
    num_words = len(palavras)
    num_unique_words = len(set(palavras))
    unique_ratio = num_unique_words / total_palavras
    num_chars = len(texto_limpo)

    num_punctuation = sum(1 for c in texto_original if c in ".,;:")
    letras = [c for c in texto_original if c.isalpha()]
    num_uppercase = sum(1 for c in letras if c.isupper()) / len(letras) if letras else 0

    num_emojis = contar_emojis(texto_original)
    num_hashtags = len(re.findall(r"#\w+", texto_original))
    num_mentions = len(re.findall(r"@\w+", texto_original))
    num_exclamation = texto_original.count("!")
    num_question = texto_original.count("?")

    return {
        "avg_word_length": avg_word_length,
        "num_words": num_words,
        "num_unique_words": num_unique_words,
        "unique_ratio": unique_ratio,
        "num_chars": num_chars,
        "num_punctuation": num_punctuation,
        "num_uppercase": num_uppercase,
        "num_emojis": num_emojis,
        "num_hashtags": num_hashtags,
        "num_mentions": num_mentions,
        "num_exclamation": num_exclamation,
        "num_question": num_question
    }


modelo = joblib.load("models/modelo_final.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")


st.set_page_config(
    page_title="AssinaturaDigital",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ AssinaturaDigital")
st.subheader("Identificação de autoria por estilo de escrita")

st.write(
    """
    Este app usa Machine Learning para tentar identificar o autor de um tweet
    com base em características de estilo, como tamanho das palavras, pontuação,
    uso de maiúsculas, emojis, exclamações e interrogações.
    """
)

tweet = st.text_area(
    "Cole aqui um tweet:",
    height=150,
    placeholder="Exemplo: The universe is under no obligation to make sense to you..."
)

if st.button("Identificar autor"):
    if len(tweet.strip()) == 0:
        st.warning("Digite ou cole um tweet antes de continuar.")
    else:
        features_extraidas = extrair_features(tweet)

        entrada = pd.DataFrame([features_extraidas])
        entrada = entrada[features]

        entrada_scaled = scaler.transform(entrada)

        autor_previsto = modelo.predict(entrada_scaled)[0]

        st.success(f"Autor mais provável: {autor_previsto}")

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(entrada_scaled)[0]
            classes = modelo.classes_

            df_probs = pd.DataFrame({
                "Autor": classes,
                "Confiança": probabilidades
            }).sort_values(by="Confiança", ascending=False)

            st.write("### Confiança por autor")
            st.dataframe(df_probs)

            st.bar_chart(df_probs.set_index("Autor"))

        st.write("### Features extraídas do texto")
        st.dataframe(entrada)