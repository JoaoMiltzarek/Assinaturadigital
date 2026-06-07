import streamlit as st
import pandas as pd
import joblib

from src.features_v1 import extrair_features_v1

from src.loaders.csv_loader import load_csv_messages
from src.loaders.generic_text_loader import load_generic_text_messages
from src.loaders.whatsapp_loader import load_whatsapp_txt_messages
from src.stylometry.profile_builder import build_author_stylometric_profile
from src.stylometry.style_replicator import rewrite_message_with_profile, explain_rewrite_rules


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400;1,600&family=Instrument+Sans:ital,wght@0,300;0,400;0,500;1,300&family=Inconsolata:wght@300;400;500&display=swap');

        :root {
            --bg:          #ede9e1;
            --bg-surface:  #e8e3da;
            --bg-card:     #ddd8ce;
            --border:      #c4bfb4;
            --border-soft: #d4cfc5;
            --text:        #1e1c18;
            --text-dim:    #6a6558;
            --text-faint:  #9e998e;
            --accent:      #1a3aff;
            --rule:        #b8b3a8;
        }

        html, body, [class*="css"] {
            font-family: 'Instrument Sans', sans-serif !important;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 820px;
            padding-top: 4rem;
            padding-bottom: 6rem;
        }

        /* ── HERO ─────────────────────────────── */
        .hero-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 32px;
            padding: 56px 0 48px;
            border-bottom: 1px solid var(--rule);
            margin-bottom: 40px;
            background: transparent;
        }

        .hero-left { flex: 1; }

        .hero-gif {
            width: 120px;
            height: 120px;
            flex-shrink: 0;
            object-fit: contain;
            opacity: 0.88;
        }

        .app-kicker { display: none; }

        .hero-title {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: clamp(3.2rem, 7vw, 5.8rem);
            line-height: 0.95;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
            color: var(--text);
        }

        .hero-title em {
            font-style: italic;
            color: var(--accent);
        }

        .hero-subtitle { display: none; }
        .meme-row      { display: none; }

        /* ── TYPOGRAPHY ───────────────────────── */
        h2 {
            font-family: 'Cormorant Garamond', Georgia, serif !important;
            font-size: 1.7rem !important;
            font-weight: 600 !important;
            color: var(--text) !important;
            letter-spacing: -0.01em !important;
            margin-top: 2rem !important;
        }

        h3 {
            font-family: 'Inconsolata', monospace !important;
            font-size: 0.72rem !important;
            font-weight: 500 !important;
            color: var(--text-faint) !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            margin-top: 1.8rem !important;
        }

        p {
            font-weight: 300;
            line-height: 1.75;
            color: var(--text-dim);
        }

        /* ── TABS ─────────────────────────────── */
        div[data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 1px solid var(--rule) !important;
            gap: 0 !important;
            margin-bottom: 36px;
        }

        button[data-baseweb="tab"] {
            border-radius: 0 !important;
            padding: 14px 24px 12px !important;
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            margin-bottom: -1px !important;
            font-family: 'Inconsolata', monospace !important;
            font-size: 0.72rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
            color: var(--text-faint) !important;
            transition: all 0.12s ease !important;
        }

        button[data-baseweb="tab"]:hover {
            color: var(--text) !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 2px solid var(--accent) !important;
        }

        div[data-baseweb="tab-highlight"] { display: none !important; }

        /* ── BUTTONS ──────────────────────────── */
        .stButton > button {
            border-radius: 0 !important;
            border: 1px solid var(--text) !important;
            background: var(--text) !important;
            color: var(--bg) !important;
            font-family: 'Inconsolata', monospace !important;
            font-size: 0.72rem !important;
            font-weight: 500 !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
            padding: 0.65rem 1.4rem !important;
            transition: all 0.12s ease !important;
        }

        .stButton > button:hover {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }

        /* ── INPUTS ───────────────────────────── */
        .stTextArea textarea,
        .stTextInput input {
            border-radius: 0 !important;
            border: 1px solid var(--border) !important;
            border-bottom: 2px solid var(--rule) !important;
            background: var(--bg-surface) !important;
            color: var(--text) !important;
            font-family: 'Instrument Sans', sans-serif !important;
            font-size: 0.95rem !important;
            line-height: 1.7 !important;
        }

        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: var(--border) !important;
            border-bottom: 2px solid var(--accent) !important;
            box-shadow: none !important;
        }

        /* ── METRICS ──────────────────────────── */
        div[data-testid="stMetric"] {
            background: var(--bg-surface);
            border: none;
            border-top: 2px solid var(--rule);
            border-radius: 0;
            padding: 18px 0 16px;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'Cormorant Garamond', Georgia, serif !important;
            font-size: 2.6rem !important;
            font-weight: 600 !important;
            color: var(--text) !important;
            letter-spacing: -0.02em !important;
            line-height: 1.0 !important;
        }

        div[data-testid="stMetricLabel"] {
            font-family: 'Inconsolata', monospace !important;
            font-size: 0.65rem !important;
            color: var(--text-faint) !important;
            font-weight: 400 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
        }

        /* ── DATAFRAMES ───────────────────────── */
        div[data-testid="stDataFrame"] {
            border-radius: 0;
            border: 1px solid var(--border) !important;
        }

        /* ── ALERTS ───────────────────────────── */
        div[data-testid="stAlert"] {
            border-radius: 0;
            border: 1px solid var(--border-soft);
            border-left: 3px solid var(--rule);
            background: var(--bg-surface) !important;
            color: var(--text-dim) !important;
        }

        /* ── CUSTOM CARDS ─────────────────────── */
        .glass-card {
            padding: 24px 28px;
            background: var(--bg-surface);
            border: 1px solid var(--border-soft);
            border-top: 2px solid var(--rule);
            margin: 12px 0 28px 0;
        }

        .section-label {
            font-family: 'Inconsolata', monospace;
            color: var(--text-faint);
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.65rem;
            margin-bottom: 12px;
        }

        .big-result {
            padding: 32px 28px 28px;
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            margin: 24px 0;
        }

        .big-result-title {
            font-family: 'Inconsolata', monospace;
            font-size: 0.65rem;
            color: var(--text-faint);
            font-weight: 400;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 10px;
        }

        .big-result-value {
            font-family: 'Cormorant Garamond', Georgia, serif;
            font-size: clamp(1.6rem, 3.5vw, 2.6rem);
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--text);
            line-height: 1.15;
        }

        .caption-soft {
            color: var(--text-dim);
            font-size: 0.88rem;
            line-height: 1.7;
            margin-top: 12px;
            font-weight: 300;
        }

        .meme-note {
            padding: 14px 18px;
            background: transparent;
            border-left: 2px solid var(--border);
            color: var(--text-faint);
            font-family: 'Inconsolata', monospace;
            font-size: 0.78rem;
            font-style: italic;
            margin: 18px 0 4px;
        }

        hr {
            border: none !important;
            border-top: 1px solid var(--rule) !important;
            margin: 36px 0 !important;
        }

        /* ── SELECT / RADIO / SLIDER ──────────── */
        div[data-baseweb="select"] > div {
            border-radius: 0 !important;
            border-color: var(--border) !important;
            background: var(--bg-surface) !important;
        }

        .stRadio label {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-size: 0.9rem !important;
            color: var(--text-dim) !important;
        }

        /* ── FILE UPLOADER ────────────────────── */
        div[data-testid="stFileUploader"] section {
            border-radius: 0 !important;
            border: 1px dashed var(--border) !important;
            background: var(--bg-warm) !important;
        }

        /* ── BAR CHART ────────────────────────── */
        div[data-testid="stVegaLiteChart"] {
            border-radius: 0;
            border: 1px solid var(--border-soft);
            padding: 16px;
            background: var(--bg-surface);
        }

        /* ── SCROLLBAR ────────────────────────── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--border); }
        </style>
        """,
        unsafe_allow_html=True
    )



modelo = joblib.load("models/modelo_final.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")


st.set_page_config(
    page_title="AssinaturaDigital",
    page_icon="✍️",
    layout="centered"
)

inject_custom_css()

st.markdown(
    """
    <div class="hero-card">
        <div class="app-kicker">Trabalho de Conclusão de Curso — Estilometria Digital</div>
        <h1 class="hero-title">Assinatura<em>Digital</em></h1>
        <div class="hero-subtitle">
            Um detector de jeitos de escrever: identifica autoria, cria perfil estilométrico
            e tenta recriar mensagens no estilo escolhido. Meio ciência de dados, meio
            fofoca estatística, meio algoritmo com crise existencial.
        </div>
        <div class="meme-row">
            <div class="pill">Machine Learning</div>
            <div class="pill">WhatsApp</div>
            <div class="pill">Escrita Digital</div>
            <div class="pill">NLP</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["01 — Classificador", "02 — Perfilador"])

with tab1:
    st.markdown("## Classificador v1")
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-label">modo detetive textual</div>
            <div class="caption-soft">
                Cole um tweet ou frase. O modelo transforma o texto em atributos numéricos
                como tamanho médio das palavras, pontuação, emojis, hashtags e maiúsculas.
                Depois tenta prever qual autor tem o estilo mais parecido.
            </div>
            <div class="meme-note">
                aviso acadêmico: se errar, não é bug existencial — texto curto é naturalmente difícil de classificar.
            </div>
        </div>
        """,
        unsafe_allow_html=True
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
            features_extraidas = extrair_features_v1(tweet)

            entrada = pd.DataFrame([features_extraidas])
            entrada = entrada[features]

            entrada_scaled = scaler.transform(entrada)

            autor_previsto = modelo.predict(entrada_scaled)[0]

            st.markdown(
                f"""
                <div class="big-result">
                    <div class="big-result-title">autor mais provável</div>
                    <div class="big-result-value">🧬 {autor_previsto}</div>
                    <div class="caption-soft">
                        O modelo acha que esse texto tem mais cara desse autor. Confie, mas desconfie.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

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
            st.dataframe(entrada)

with tab2:
    st.markdown("## Perfilador v2")
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-label">raio-x de conversa</div>
            <div class="caption-soft">
                Envie uma conversa completa, selecione o autor e gere um perfil de escrita:
                palavras frequentes, bigramas, risadas, emojis, interrogações e exemplos reais.
                Depois o sistema tenta recriar uma mensagem no estilo detectado.
            </div>
            <div class="meme-note">
                energia: “li 437 mensagens e agora acho que entendi sua personalidade”.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if "v2_profile" not in st.session_state:
        st.session_state["v2_profile"] = None

    if st.session_state["v2_profile"] is None:
        input_type = st.radio(
            "Tipo de entrada:",
            ["Texto simples", "CSV", "WhatsApp (.txt)"],
        )

        df = None

        if input_type == "Texto simples":
            author_name = st.text_input("Nome do autor:", "Joao")
            raw_text = st.text_area(
                "Cole as mensagens aqui, uma por linha:",
                height=180,
                placeholder="oi\nkkkk sério isso?\nbora amanhã\nnão sei... talvez"
            )
            if raw_text.strip():
                df = load_generic_text_messages(raw_text, author_name=author_name)

        elif input_type == "CSV":
            uploaded_file = st.file_uploader(
                "Envie um arquivo CSV com mensagens",
                type=["csv"]
            )
            if uploaded_file is not None:
                temp_df = pd.read_csv(uploaded_file)
                st.write("Colunas disponíveis no CSV:")
                st.dataframe(pd.DataFrame({"colunas": list(temp_df.columns)}))

                col1, col2 = st.columns(2)
                with col1:
                    author_col = st.selectbox("Coluna de Autor", temp_df.columns)
                with col2:
                    text_col = st.selectbox("Coluna de Texto", temp_df.columns)

                if st.button("Carregar CSV"):
                    uploaded_file.seek(0)
                    df = load_csv_messages(uploaded_file, author_column=author_col, text_column=text_col)

        elif input_type == "WhatsApp (.txt)":
            uploaded_file = st.file_uploader(
                "Envie o .txt exportado do WhatsApp",
                type=["txt"]
            )
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
                st.session_state["v2_profile"] = profile
                st.rerun()

    else:
        profile = st.session_state["v2_profile"]

        st.markdown(
            f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;text-transform:uppercase;
            letter-spacing:0.12em;color:#a8a49c;margin-bottom:6px">perfil gerado</div>
            <div style="font-family:'Libre Baskerville',Georgia,serif;font-size:1.6rem;font-weight:700;
            letter-spacing:-0.02em;color:#1a1916;margin-bottom:28px">{profile['author']}</div>
            """,
            unsafe_allow_html=True
        )

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Mensagens analisadas", profile["message_count"])
        col_m2.metric("Média de palavras", f"{profile['avg_words_per_message']:.1f}")
        col_m3.metric("Total de risadas", int(profile["total_laughter"]))

        col_m4, col_m5, col_m6 = st.columns(3)
        col_m4.metric("Total de emojis", int(profile["total_emojis"]))
        col_m5.metric("Total de exclamações", int(profile["total_exclamation"]))
        col_m6.metric("Total de interrogações", int(profile["total_question"]))

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.write("### Palavras frequentes")
            st.dataframe(pd.DataFrame(profile["top_words"]))
        with col_t2:
            st.write("### Bigramas frequentes")
            st.dataframe(pd.DataFrame(profile["top_bigrams"]))

        samples = profile["sample_messages"]
        st.markdown("### Exemplos reais")
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex;flex-direction:column;gap:16px;">
                <div>
                    <div class="section-label">mensagem curta</div>
                    <div style="color:#1a1916;font-size:0.95rem;line-height:1.7;font-family:'IBM Plex Sans',sans-serif">{samples.get('short', 'N/A')}</div>
                </div>
                <div style="border-top:1px solid #e8e5de;padding-top:16px">
                    <div class="section-label">mensagem média</div>
                    <div style="color:#1a1916;font-size:0.95rem;line-height:1.7;font-family:'IBM Plex Sans',sans-serif">{samples.get('medium', 'N/A')}</div>
                </div>
                <div style="border-top:1px solid #e8e5de;padding-top:16px">
                    <div class="section-label">mensagem longa</div>
                    <div style="color:#1a1916;font-size:0.95rem;line-height:1.7;font-family:'IBM Plex Sans',sans-serif">{samples.get('long', 'N/A')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.write("## Recriar no estilo do autor")
        st.write(
            """
            Digite uma mensagem neutra e o sistema tentará recriá-la
            usando características do autor selecionado.
            """
        )

        neutral_message = st.text_area(
            "Mensagem base:",
            height=120,
            placeholder="Exemplo: hoje tem reunião às 14h"
        )

        intensity = st.slider(
            "Intensidade da recriação:",
            min_value=1,
            max_value=3,
            value=2,
            help="1 = leve, 2 = normal, 3 = exagerado"
        )

        if st.button("Recriar no estilo", key="recriar_estilo_v2"):
            if not neutral_message.strip():
                st.warning("Digite uma mensagem base")
            else:
                rewritten = rewrite_message_with_profile(
                    neutral_message,
                    profile,
                    intensity=intensity
                )

                st.markdown("### 🧾 Mensagem original")
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div class="caption-soft">{neutral_message}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 🪄 Mensagem recriada")
                st.markdown(
                    f"""
                    <div class="big-result">
                        <div class="big-result-title">saída estilométrica</div>
                        <div class="big-result-value">{rewritten}</div>
                        <div class="caption-soft">
                            Reescrita baseada no perfil detectado. Não é possessão digital, é regra + estatística.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("Regras usadas na recriação")
                for rule in explain_rewrite_rules(profile):
                    st.write(f"- {rule}")

        if st.button("Limpar perfil atual", key="limpar_perfil_v2"):
            st.session_state["v2_profile"] = None
            st.rerun()