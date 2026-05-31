"""
src/stylometry/profile_builder.py

Construtor de perfil estilométrico agregado para um autor específico.

Responsabilidade única: dado um DataFrame com múltiplas mensagens e o nome
de um autor, calcular médias, totais e análises textuais, retornando um
dicionário completo de perfil estilométrico.

Versão 4B — adicionadas as chaves:
    top_words      : palavras mais frequentes do autor
    top_bigrams    : bigramas (pares de palavras) mais frequentes
    sample_messages: exemplos real de mensagens (curta, média, longa)
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.processing.cleaner import clean_text_for_stylometry
from src.stylometry.feature_extractor import extract_stylometric_features

# ---------------------------------------------------------------------------
# Colunas obrigatórias no DataFrame de entrada
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS = {"author", "text"}

# ---------------------------------------------------------------------------
# Stopwords básicas para filtrar palavras muito comuns na contagem de termos.
# Propositalmente pequena — bigramas NÃO usam esta lista.
# ---------------------------------------------------------------------------
_STOPWORDS: set[str] = {
    # Português
    "a", "o", "e", "é", "de", "do", "da", "em", "no", "na", "os", "as",
    "um", "uma", "uns", "umas", "que", "se", "me", "te", "lhe", "nos",
    "com", "por", "para", "mas", "ou", "pra", "pro", "num", "numa",
    "eu", "tu", "ele", "ela", "nós", "vocês", "eles", "elas", "você",
    "já", "mais", "menos", "não", "sim", "meu", "minha", "seu", "sua",
    "isso", "esta", "esse", "este", "esse", "aqui", "ali", "lá",
    "foi", "era", "ser", "ter", "vai", "vou", "tem", "são", "está",
    # Inglês
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "of", "for", "is", "it", "its", "be", "by", "as", "at", "so",
    "do", "my", "me", "we", "us", "he", "she", "they", "this", "that",
    "i", "you", "he", "she", "we", "they", "am", "are", "was", "were",
}

# Máximo de termos/bigramas retornados
_TOP_N = 10

# Mínimo de caracteres para uma palavra ser contada (filtra "a", "e", etc.)
_MIN_WORD_LENGTH = 2


# ---------------------------------------------------------------------------
# Funções privadas de suporte
# ---------------------------------------------------------------------------

def _tokenize_words_for_profile(text: str) -> list[str]:
    """
    Tokeniza um texto limpo em palavras minúsculas válidas para contagem.

    Filtra tokens curtos e stopwords. Usado exclusivamente para top_words.
    """
    words = text.lower().split()
    return [
        w for w in words
        if len(w) >= _MIN_WORD_LENGTH and w not in _STOPWORDS
    ]


def _tokenize_all_words(text: str) -> list[str]:
    """
    Tokeniza um texto limpo em palavras minúsculas SEM filtrar stopwords.

    Usado para construir bigramas, onde expressões como "pior que" são
    importantes no português informal.
    """
    return text.lower().split()


def _build_top_words(cleaned_texts: list[str]) -> list[dict[str, object]]:
    """
    Retorna as palavras mais frequentes (sem stopwords) entre todas as mensagens.

    Args:
        cleaned_texts: Lista de textos já limpos com clean_text_for_stylometry.

    Returns:
        Lista de dicts [{"term": str, "count": int}] ordenada por frequência.
    """
    counter: Counter[str] = Counter()
    for text in cleaned_texts:
        tokens = _tokenize_words_for_profile(text)
        counter.update(tokens)

    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(_TOP_N)
    ]


def _build_top_bigrams(cleaned_texts: list[str]) -> list[dict[str, object]]:
    """
    Retorna os bigramas (pares de palavras consecutivas) mais frequentes.

    NÃO remove stopwords para preservar expressões informais importantes
    no português como "pior que", "tipo assim", "muito bom".

    Args:
        cleaned_texts: Lista de textos já limpos com clean_text_for_stylometry.

    Returns:
        Lista de dicts [{"term": str, "count": int}] ordenada por frequência.
    """
    counter: Counter[str] = Counter()
    for text in cleaned_texts:
        tokens = _tokenize_all_words(text)
        # Cria pares (bigrams) de palavras consecutivas
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        counter.update(bigrams)

    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(_TOP_N)
    ]


def _select_sample_messages(messages: list[str]) -> dict[str, str]:
    """
    Seleciona exemplos reais de mensagens: curta, média e longa.

    - short  = menor mensagem não vazia (por número de caracteres)
    - long   = maior mensagem não vazia (por número de caracteres)
    - medium = mensagem cujo tamanho é mais próximo da mediana

    Se houver poucas mensagens, uma mesma mensagem pode aparecer em
    mais de uma categoria.

    Args:
        messages: Lista de textos originais (não limpos) do autor.

    Returns:
        Dicionário com chaves 'short', 'medium' e 'long'.
    """
    # Filtra mensagens vazias
    non_empty = [m for m in messages if m and str(m).strip()]
    if not non_empty:
        return {"short": "", "medium": "", "long": ""}

    # Ordena por tamanho (número de caracteres)
    sorted_by_length = sorted(non_empty, key=lambda m: len(str(m)))

    short = sorted_by_length[0]
    long = sorted_by_length[-1]

    # Mediana: índice central
    median_index = len(sorted_by_length) // 2
    medium = sorted_by_length[median_index]

    return {
        "short": str(short),
        "medium": str(medium),
        "long": str(long),
    }


# ---------------------------------------------------------------------------
# Função pública principal
# ---------------------------------------------------------------------------

def build_author_stylometric_profile(
    messages_dataframe: pd.DataFrame,
    author_name: str,
) -> dict[str, object]:
    """
    Constrói um perfil estilométrico completo e agregado para um autor.

    Filtra as mensagens do autor no DataFrame, extrai features numéricas
    de cada mensagem, agrega em médias e totais, e adiciona análise de
    vocabulário (top_words, top_bigrams) e exemplos reais (sample_messages).

    Args:
        messages_dataframe: DataFrame com pelo menos as colunas 'author' e 'text'.
        author_name: Nome exato do autor cujo perfil será construído.

    Returns:
        Dicionário com as chaves:
            author, message_count,
            avg_words_per_message, avg_chars_per_message, avg_word_length,
            avg_unique_ratio, avg_uppercase_ratio, avg_punctuation_ratio,
            avg_emoji_ratio, total_emojis, total_laughter,
            total_exclamation, total_question,
            top_words, top_bigrams, sample_messages

    Raises:
        ValueError: Se as colunas obrigatórias não existirem.
        ValueError: Se o autor não tiver mensagens no DataFrame.
    """
    # ------------------------------------------------------------------
    # Validação das colunas obrigatórias
    # ------------------------------------------------------------------
    missing_columns = _REQUIRED_COLUMNS - set(messages_dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"O DataFrame não contém as colunas obrigatórias: {missing_columns}. "
            f"Colunas encontradas: {list(messages_dataframe.columns)}"
        )

    # ------------------------------------------------------------------
    # Filtro: apenas mensagens do autor solicitado
    # ------------------------------------------------------------------
    author_rows: pd.DataFrame = messages_dataframe[
        messages_dataframe["author"] == author_name
    ]

    if author_rows.empty:
        available = messages_dataframe["author"].unique().tolist()
        raise ValueError(
            f"Nenhuma mensagem encontrada para o autor '{author_name}'. "
            f"Autores disponíveis: {available}"
        )

    # Lista de textos originais e textos limpos
    raw_texts: list[str] = [str(t) for t in author_rows["text"]]
    cleaned_texts: list[str] = [clean_text_for_stylometry(t) for t in raw_texts]

    # ------------------------------------------------------------------
    # Extração de features numéricas por mensagem (não altera lógica 4A)
    # ------------------------------------------------------------------
    all_features: list[dict[str, float | int]] = [
        extract_stylometric_features(t) for t in raw_texts
    ]

    message_count: int = len(all_features)

    def _avg(key: str) -> float:
        return sum(f[key] for f in all_features) / message_count

    def _total(key: str) -> int:
        return sum(f[key] for f in all_features)

    # ------------------------------------------------------------------
    # Montagem do perfil (4A + 4B)
    # ------------------------------------------------------------------
    profile: dict[str, object] = {
        # --- Métricas numéricas (4A) ---
        "author": author_name,
        "message_count": message_count,
        "avg_words_per_message": _avg("num_words"),
        "avg_chars_per_message": _avg("num_chars"),
        "avg_word_length": _avg("avg_word_length"),
        "avg_unique_ratio": _avg("unique_ratio"),
        "avg_uppercase_ratio": _avg("uppercase_ratio"),
        "avg_punctuation_ratio": _avg("punctuation_ratio"),
        "avg_emoji_ratio": _avg("emoji_ratio"),
        "total_emojis": _total("num_emojis"),
        "total_laughter": _total("total_laughter"),
        "total_exclamation": _total("num_exclamation"),
        "total_question": _total("num_question"),

        # --- Análise de vocabulário (4B) ---
        "top_words": _build_top_words(cleaned_texts),
        "top_bigrams": _build_top_bigrams(cleaned_texts),

        # --- Exemplos reais (4B) ---
        "sample_messages": _select_sample_messages(raw_texts),
    }

    return profile
