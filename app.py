import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import emoji
import matplotlib.pyplot as plt
import io

from src.loaders.csv_loader import load_csv_messages
from src.loaders.generic_text_loader import load_generic_text_messages
from src.loaders.whatsapp_loader import load_whatsapp_txt_messages
from src.stylometry.profile_builder import build_author_stylometric_profile


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

tab1, tab2 = st.tabs(["Classificador v1", "Perfilador v2"])

with tab1:
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

with tab2:
    st.subheader("Perfilador Estilométrico v2")
    st.write("Envie mensagens para criar um perfil estilométrico de uma pessoa.")
    
    input_type = st.radio("Escolha o tipo de entrada:", ["Texto simples", "CSV", "WhatsApp TXT"])
    
    df = None
    
    if input_type == "Texto simples":
        author_name = st.text_input("Nome do autor:", "Joao")
        raw_text = st.text_area("Cole as mensagens (uma por linha):", height=150)
        if raw_text.strip():
            df = load_generic_text_messages(raw_text, author_name=author_name)
            
    elif input_type == "CSV":
        uploaded_file = st.file_uploader("Envie um arquivo CSV", type=["csv"])
        if uploaded_file is not None:
            temp_df = pd.read_csv(uploaded_file)
            st.write("Colunas disponíveis:", list(temp_df.columns))
            
            col1, col2 = st.columns(2)
            with col1:
                author_col = st.selectbox("Coluna de Autor", temp_df.columns)
            with col2:
                text_col = st.selectbox("Coluna de Texto", temp_df.columns)
                
            if st.button("Carregar CSV"):
                uploaded_file.seek(0)
                df = load_csv_messages(uploaded_file, author_column=author_col, text_column=text_col)
                
    elif input_type == "WhatsApp TXT":
        uploaded_file = st.file_uploader("Envie o .txt exportado do WhatsApp", type=["txt"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8", errors="replace")
            df = load_whatsapp_txt_messages(content)
            
    if df is not None and not df.empty:
        st.success(f"{len(df)} mensagens carregadas com sucesso.")
        autores = df["author"].value_counts()
        st.write("Mensagens por autor:")
        st.dataframe(autores)
        
        selected_author = st.selectbox("Selecione o autor para gerar o perfil:", autores.index)
        
        if st.button("Gerar Perfil"):
            profile = build_author_stylometric_profile(df, selected_author)
            
            st.write(f"### Perfil: {profile['author']}")
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Mensagens Analisadas", profile["message_count"])
            col_m2.metric("Total de Risadas", profile["total_laughter"])
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.write("#### Palavras mais frequentes")
                st.dataframe(pd.DataFrame(profile["top_words"]))
            with col_t2:
                st.write("#### Bigramas mais frequentes")
                st.dataframe(pd.DataFrame(profile["top_bigrams"]))
                
            st.write("#### Exemplos de Mensagens")
            samples = profile["sample_messages"]
            st.info(f"**Curta:** {samples['short']}")
            st.info(f"**Média:** {samples['medium']}")
            st.info(f"**Longa:** {samples['long']}")