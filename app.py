import streamlit as st
import pandas as pd
import joblib

from src.features_v1 import extrair_features_v1
from src.loaders.csv_loader import load_csv_messages
from src.loaders.generic_text_loader import load_generic_text_messages
from src.loaders.whatsapp_loader import load_whatsapp_txt_messages
from src.stylometry.profile_builder import build_author_stylometric_profile
from src.stylometry.style_replicator import (
    rewrite_message_with_profile,
    explain_rewrite_rules,
)


# ============================================================
# Configuração visual
# ============================================================

st.set_page_config(
    page_title="AssinaturaDigital",
    page_icon="✍️",
    layout="centered",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Instrument+Sans:wght@300;400;500;600&family=Inconsolata:wght@400;500&display=swap');

        :root {
            --bg: #ede9e1;
            --surface: #e8e3da;
            --card: #ddd8ce;
            --border: #c4bfb4;
            --text: #1e1c18;
            --muted: #6a6558;
            --faint: #9e998e;
            --accent: #1a3aff;
        }

        html, body, [class*="css"] {
            font-family: 'Instrument Sans', sans-serif !important;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: 900px;
            padding-top: 3rem;
            padding-bottom: 5rem;
        }

        h1, h2 {
            font-family: 'Cormorant Garamond', Georgia, serif !important;
            letter-spacing: -0.02em !important;
        }

        h1 {
            font-size: clamp(3rem, 7vw, 5rem) !important;
            line-height: 0.95 !important;
            margin-bottom: 0.5rem !important;
        }

        h2 {
            font-size: 2rem !important;
            margin-top: 2rem !important;
        }

        h3 {
            font-family: 'Inconsolata', monospace !important;
            color: var(--faint) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            font-size: 0.8rem !important;
        }

        .hero {
            border-bottom: 1px solid var(--border);
            padding-bottom: 2rem;
            margin-bottom: 2rem;
        }

        .hero em {
            color: var(--accent);
            font-style: italic;
        }

        .subtitle {
            color: var(--muted);
            line-height: 1.7;
            max-width: 760px;
        }

        .info-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            padding: 1.2rem 1.4rem;
            margin: 1rem 0 1.5rem;
        }

        .soft-note {
            font-family: 'Inconsolata', monospace;
            color: var(--faint);
            font-size: 0.85rem;
        }

        .stButton > button {
            border-radius: 0 !important;
            border: 1px solid var(--text) !important;
            background: var(--text) !important;
            color: var(--bg) !important;
            font-family: 'Inconsolata', monospace !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
        }

        .stButton > button:hover {
            border-color: var(--accent) !important;
            background: var(--accent) !important;
            color: white !important;
        }

        .stTextArea textarea,
        .stTextInput input {
            border-radius: 0 !important;
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
        }

        div[data-baseweb="select"] > div {
            border-radius: 0 !important;
            background: var(--surface) !important;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# ============================================================
# Funções auxiliares
# ============================================================

@st.cache_resource
def load_v1_pipeline():
    """
    Carrega o pipeline único da V1.

    Ele deve conter:
    - features manuais;
    - TF-IDF char n-grams;
    - scaler, se houver;
    - classificador final.
    """
    return joblib.load("models/v1_pipeline.pkl")


def load_whatsapp_with_optional_stats(content: str):
    """
    Carrega WhatsApp de forma compatível com duas versões do loader:

    1. Versão nova:
       load_whatsapp_txt_messages(content, return_stats=True)

    2. Versão antiga:
       load_whatsapp_txt_messages(content)

    Assim o app não quebra se o loader ainda não tiver estatísticas.
    """
    try:
        df, stats = load_whatsapp_txt_messages(content, return_stats=True)
        return df, stats

    except TypeError:
        df = load_whatsapp_txt_messages(content)
        stats = {
            "total_records_found": len(df),
            "valid_messages": len(df),
            "discarded_messages": 0,
            "discard_reasons": {},
            "warning": (
                "O loader atual não retornou estatísticas de descarte. "
                "A conversa foi carregada normalmente, mas sem resumo detalhado."
            ),
        }
        return df, stats


def get_preview_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["datetime", "author", "text", "source"]
    return [column for column in preferred if column in df.columns]


def render_messages_loaded_area(df: pd.DataFrame) -> None:
    st.success(f"{len(df)} mensagens carregadas com sucesso.")

    preview_columns = get_preview_columns(df)

    if preview_columns:
        st.write("### Prévia das mensagens limpas")
        st.caption("Estas são as mensagens que realmente entrarão no perfil.")
        st.dataframe(df[preview_columns].head(20), use_container_width=True)

    st.write("### Mensagens por autor")
    autores = df["author"].value_counts()
    st.dataframe(autores, use_container_width=True)

    if len(df) < 10:
        st.warning(
            "Há poucas mensagens carregadas. O perfil pode ficar frágil. "
            "Para uma análise melhor, use mais exemplos."
        )


def render_profile(profile: dict) -> None:
    author = profile.get("author", "autor desconhecido")

    st.write("## Perfil gerado")
    st.subheader(str(author))

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Mensagens analisadas", int(profile.get("message_count", 0)))
    col_m2.metric("Média de palavras", f"{float(profile.get('avg_words_per_message', 0)):.1f}")
    col_m3.metric("Total de risadas", int(profile.get("total_laughter", 0)))

    col_m4, col_m5, col_m6 = st.columns(3)
    col_m4.metric("Total de emojis", int(profile.get("total_emojis", 0)))
    col_m5.metric("Exclamações", int(profile.get("total_exclamation", 0)))
    col_m6.metric("Interrogações", int(profile.get("total_question", 0)))

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.write("### Palavras frequentes")
        top_words = profile.get("top_words", [])
        if top_words:
            st.dataframe(pd.DataFrame(top_words), use_container_width=True)
        else:
            st.info("Nenhuma palavra frequente encontrada.")

    with col_t2:
        st.write("### Bigramas frequentes")
        top_bigrams = profile.get("top_bigrams", [])
        if top_bigrams:
            st.dataframe(pd.DataFrame(top_bigrams), use_container_width=True)
        else:
            st.info("Nenhum bigrama frequente encontrado.")

    detected_abbreviations = profile.get("detected_abbreviations", {})
    if detected_abbreviations:
        st.write("### Abreviações detectadas")
        st.dataframe(
            pd.DataFrame(
                [
                    {"abreviação": key, "quantidade": value}
                    for key, value in detected_abbreviations.items()
                ]
            ),
            use_container_width=True,
        )

    laughter_patterns = profile.get("laughter_patterns", {})
    if laughter_patterns:
        st.write("### Risadas detectadas")
        st.dataframe(
            pd.DataFrame(
                [
                    {"risada": key, "quantidade": value}
                    for key, value in laughter_patterns.items()
                ]
            ),
            use_container_width=True,
        )

    samples = profile.get("sample_messages", {}) or {}

    st.write("### Exemplos reais usados como referência")
    st.write("**Mensagem curta:**")
    st.info(samples.get("short", "N/A"))

    st.write("**Mensagem média:**")
    st.info(samples.get("medium", "N/A"))

    st.write("**Mensagem longa:**")
    st.info(samples.get("long", "N/A"))

    st.divider()

    st.write("## Recriar no estilo do autor")

    neutral_message = st.text_area(
        "Mensagem base:",
        height=120,
        placeholder="Exemplo: hoje tem reunião às 14h",
        key="neutral_message_v2",
    )

    intensity = st.slider(
        "Intensidade da recriação:",
        min_value=1,
        max_value=3,
        value=2,
        help="1 = leve, 2 = normal, 3 = mais forte",
        key="intensity_v2",
    )

    if st.button("Recriar no estilo", key="recriar_estilo_v2"):
        if not neutral_message.strip():
            st.warning("Digite uma mensagem base.")
        else:
            rewritten = rewrite_message_with_profile(
                neutral_message,
                profile,
                intensity=intensity,
            )

            st.write("### Mensagem original")
            st.info(neutral_message)

            st.write("### Mensagem recriada")
            st.success(rewritten)

            st.write("### Regras usadas")
            for rule in explain_rewrite_rules(profile):
                st.write(f"- {rule}")


def reset_v2_profile() -> None:
    st.session_state["v2_profile"] = None


def reset_v2_messages() -> None:
    st.session_state["v2_messages_df"] = None


# ============================================================
# Cabeçalho
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Assinatura<em>Digital</em></h1>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Estado da sessão
# ============================================================

if "v2_profile" not in st.session_state:
    st.session_state["v2_profile"] = None

if "v2_messages_df" not in st.session_state:
    st.session_state["v2_messages_df"] = None

if "v2_last_input_type" not in st.session_state:
    st.session_state["v2_last_input_type"] = None


# ============================================================
# Abas
# ============================================================

tab1, tab2 = st.tabs(["01 — Classificador", "02 — Perfilador"])


# ============================================================
# Aba V1
# ============================================================

with tab1:
    st.markdown("## Classificador V1")

    st.markdown(
    """
    <div class="info-card">
        Classifica autoria de um texto com base em características de escrita.
    </div>
    """,
    unsafe_allow_html=True,
)


    tweet = st.text_area(
        "Cole aqui um tweet ou frase:",
        height=150,
        placeholder="Exemplo: The universe is under no obligation to make sense to you...",
        key="tweet_v1",
    )

    if st.button("Identificar autor", key="identificar_autor_v1"):
        if not tweet.strip():
            st.warning("Digite ou cole um tweet antes de continuar.")
        else:
            try:
                pipeline_v1 = load_v1_pipeline()
            except FileNotFoundError:
                st.error(
                    "Pipeline da V1 não encontrado. Rode primeiro no PowerShell: "
                    "python scripts\\train_v1_pipeline.py"
                )
                st.stop()

            features_extraidas = extrair_features_v1(tweet)
            entrada = pd.DataFrame([features_extraidas])

            autor_previsto = pipeline_v1.predict([tweet])[0]

            st.success(f"Autor mais provável: {autor_previsto}")

            if hasattr(pipeline_v1, "predict_proba"):
                probabilidades = pipeline_v1.predict_proba([tweet])[0]
                classes = pipeline_v1.classes_

                df_probs = pd.DataFrame(
                    {
                        "Autor": classes,
                        "Confiança": probabilidades,
                    }
                ).sort_values(by="Confiança", ascending=False)

                st.write("### Confiança por autor")
                st.dataframe(df_probs, use_container_width=True)
                st.bar_chart(df_probs.set_index("Autor"))

            elif hasattr(pipeline_v1, "decision_function"):
                scores = pipeline_v1.decision_function([tweet])[0]
                classes = pipeline_v1.classes_

                df_scores = pd.DataFrame(
                    {
                        "Autor": classes,
                        "Score interno": scores,
                    }
                ).sort_values(by="Score interno", ascending=False)

                st.write("### Ranking interno por autor")
                st.dataframe(df_scores, use_container_width=True)
                st.bar_chart(df_scores.set_index("Autor"))

            st.write("### Features manuais extraídas")
            st.dataframe(entrada, use_container_width=True)


# ============================================================
# Aba V2
# ============================================================

with tab2:
    st.markdown("## Perfilador V2")

    st.markdown(
    """
    <div class="info-card">
        Gera um perfil de escrita a partir de mensagens do autor selecionado.
    </div>
    """,
    unsafe_allow_html=True,
)


    if st.session_state["v2_profile"] is None:
        input_type = st.radio(
            "Tipo de entrada:",
            ["Texto simples", "CSV", "WhatsApp (.txt)"],
            key="v2_input_type",
        )

        if st.session_state["v2_last_input_type"] != input_type:
            st.session_state["v2_messages_df"] = None
            st.session_state["v2_last_input_type"] = input_type

        # ----------------------------------------------------
        # Entrada por texto simples
        # ----------------------------------------------------
        if input_type == "Texto simples":
            author_name = st.text_input(
                "Nome do autor:",
                "Joao",
                key="generic_author_name",
            )

            raw_text = st.text_area(
                "Cole as mensagens aqui, uma por linha:",
                height=180,
                placeholder="oi\nkkkk sério isso?\nbora amanhã\nnão sei... talvez",
                key="generic_raw_text",
            )

            if raw_text.strip():
                df_text = load_generic_text_messages(
                    raw_text,
                    author_name=author_name,
                )
                st.session_state["v2_messages_df"] = df_text

        # ----------------------------------------------------
        # Entrada por CSV
        # ----------------------------------------------------
        elif input_type == "CSV":
            uploaded_csv = st.file_uploader(
                "Envie um arquivo CSV com mensagens",
                type=["csv"],
                key="csv_uploader_v2",
            )

            if uploaded_csv is not None:
                try:
                    preview_df = pd.read_csv(uploaded_csv)
                    uploaded_csv.seek(0)
                except Exception as exc:
                    st.error(f"Não consegui ler o CSV: {exc}")
                    preview_df = None

                if preview_df is not None:
                    st.write("### Prévia do CSV")
                    st.dataframe(preview_df.head(10), use_container_width=True)

                    columns = preview_df.columns.tolist()

                    if not columns:
                        st.error("O CSV não tem colunas.")
                    else:
                        col1, col2 = st.columns(2)

                        with col1:
                            author_column = st.selectbox(
                                "Coluna do autor",
                                columns,
                                key="csv_author_column",
                            )

                        with col2:
                            text_column = st.selectbox(
                                "Coluna do texto",
                                columns,
                                key="csv_text_column",
                            )

                        datetime_column = st.selectbox(
                            "Coluna de data/hora, se existir",
                            ["Nenhuma"] + columns,
                            key="csv_datetime_column",
                        )

                        if st.button("Carregar CSV", key="carregar_csv_v2"):
                            datetime_column_value = None

                            if datetime_column != "Nenhuma":
                                datetime_column_value = datetime_column

                            uploaded_csv.seek(0)

                            try:
                                df_csv = load_csv_messages(
                                    uploaded_csv,
                                    author_column=author_column,
                                    text_column=text_column,
                                    datetime_column=datetime_column_value,
                                    source_name="csv_upload",
                                )

                                st.session_state["v2_messages_df"] = df_csv
                                st.success(f"{len(df_csv)} mensagens carregadas do CSV.")

                            except Exception as exc:
                                st.error(f"Erro ao carregar CSV: {exc}")

        # ----------------------------------------------------
        # Entrada por WhatsApp
        # ----------------------------------------------------
        elif input_type == "WhatsApp (.txt)":
            uploaded_file = st.file_uploader(
                "Envie o .txt exportado do WhatsApp",
                type=["txt"],
                key="whatsapp_uploader_v2",
            )

            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8", errors="replace")

                try:
                    df_whatsapp, import_stats = load_whatsapp_with_optional_stats(content)
                    st.session_state["v2_messages_df"] = df_whatsapp

                    st.write("### Estatísticas de importação")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Registros encontrados", import_stats["total_records_found"])
                    col_b.metric("Mensagens válidas", import_stats["valid_messages"])
                    col_c.metric("Mensagens descartadas", import_stats["discarded_messages"])

                    if import_stats.get("warning"):
                        st.warning(import_stats["warning"])

                    discard_reasons = import_stats.get("discard_reasons", {})
                    if discard_reasons:
                        st.write("Motivos de descarte")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {"motivo": key, "quantidade": value}
                                    for key, value in discard_reasons.items()
                                ]
                            ),
                            use_container_width=True,
                        )

                except Exception as exc:
                    st.error(f"Erro ao carregar conversa do WhatsApp: {exc}")

        # ----------------------------------------------------
        # Área comum depois do carregamento
        # ----------------------------------------------------
        df = st.session_state.get("v2_messages_df")

        if df is not None:
            if df.empty:
                st.warning("Nenhuma mensagem válida foi carregada.")
            else:
                render_messages_loaded_area(df)

                autores = df["author"].value_counts()

                selected_author = st.selectbox(
                    "Selecione o autor para gerar o perfil:",
                    autores.index.tolist(),
                    key="selected_author_v2",
                )

                if st.button("Gerar Perfil", key="gerar_perfil_v2"):
                    try:
                        profile = build_author_stylometric_profile(df, selected_author)
                        st.session_state["v2_profile"] = profile
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Erro ao gerar perfil: {exc}")

        col_clear_1, col_clear_2 = st.columns(2)

        with col_clear_1:
            if st.button("Limpar mensagens carregadas", key="limpar_mensagens_v2"):
                reset_v2_messages()
                st.rerun()

    else:
        profile = st.session_state["v2_profile"]

        render_profile(profile)

        if st.button("Limpar perfil atual", key="limpar_perfil_v2"):
            reset_v2_profile()
            st.rerun()