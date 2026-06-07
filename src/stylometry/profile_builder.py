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
import re

import pandas as pd

from src.processing.cleaner import clean_text_for_stylometry
from src.stylometry.feature_extractor import extract_stylometric_features


_REQUIRED_COLUMNS = {"author", "text"}


_STOPWORDS: set[str] = {

    "a", "o", "e", "é", "de", "do", "da", "em", "no", "na", "os", "as",
    "um", "uma", "uns", "umas", "que", "se", "me", "te", "lhe", "nos",
    "com", "por", "para", "mas", "ou", "pra", "pro", "num", "numa",
    "eu", "tu", "ele", "ela", "nós", "vocês", "eles", "elas", "você",
    "já", "mais", "menos", "não", "sim", "meu", "minha", "seu", "sua",
    "isso", "esta", "esse", "este", "esse", "aqui", "ali", "lá",
    "foi", "era", "ser", "ter", "vai", "vou", "tem", "são", "está",

    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "of", "for", "is", "it", "its", "be", "by", "as", "at", "so",
    "do", "my", "me", "we", "us", "he", "she", "they", "this", "that",
    "i", "you", "he", "she", "we", "they", "am", "are", "was", "were",
}


_TOP_N = 10


_MIN_WORD_LENGTH = 2




def _normalize_profile_token(token: str) -> str:
    """
    Normaliza uma palavra para contagem no perfil.
    Remove datas, horários, mídia omitida, links e marcadores técnicos.
    """
    token = str(token).lower().strip()
    token = token.strip("[](){}.,;:!?\"'“”‘’<>")

    if not token:
        return ""

    # Datas: 7/1/26, 10/2/2026, 07-01-26, 07.01.26
    if re.fullmatch(r"\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}", token):
        return ""

    # Horários: 12:21 ou 12:21:05
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", token):
        return ""

    # Links
    if token.startswith("http") or token.startswith("www"):
        return ""

    # Lixo técnico de WhatsApp
    noise_tokens = {
        "omitted",
        "sticker",
        "audio",
        "image",
        "video",
        "gif",
        "document",
        "location",
        "contact",
        "card",
        "voice",
        "call",
        "missed",
        "this",
        "message",
        "was",
        "edited",
        "deleted",
        "mídia",
        "midia",
        "oculta",
        "imagem",
        "áudio",
        "audio",
        "vídeo",
        "video",
        "figurinha",
        "omitida",
        "omitido",
    }

    if token in noise_tokens:
        return ""


    if token.isdigit():
        return ""

    return token


def _tokenize_words_for_profile(text: str) -> list[str]:
    """
    Tokeniza palavras para top_words.

    Remove:
    - stopwords;
    - datas;
    - horários;
    - mídia omitida;
    - tokens técnicos do WhatsApp.
    """
    raw_words = str(text).lower().split()
    words = []

    for raw_word in raw_words:
        word = _normalize_profile_token(raw_word)

        if len(word) >= _MIN_WORD_LENGTH and word not in _STOPWORDS:
            words.append(word)

    return words


def _tokenize_all_words(text: str) -> list[str]:
    """
    Tokeniza palavras para bigramas.

    Mantém stopwords porque expressões como "to aqui", "na academia",
    "pior que" podem ser estilo real.
    """
    raw_words = str(text).lower().split()
    words = []

    for raw_word in raw_words:
        word = _normalize_profile_token(raw_word)

        if len(word) >= _MIN_WORD_LENGTH:
            words.append(word)

    return words


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

        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        counter.update(bigrams)

    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(_TOP_N)
    ]


def _is_valid_sample_message(message: str) -> bool:
    """Evita usar mensagens vazias, pontuação solta ou mídia omitida como exemplo."""
    text = str(message).strip()
    lower = text.lower()

    if len(text) < 3:
        return False

    if re.fullmatch(r"[\W_]+", text):
        return False

    if "omitted" in lower and len(lower.split()) <= 4:
        return False

    if lower in {"sticker omitted", "audio omitted", "image omitted", "video omitted"}:
        return False

    return True


def _is_valid_sample_message(message: str) -> bool:
    """
    Evita usar mídia, mensagem apagada/editada, chamada ou lixo como exemplo.
    """
    text = str(message).strip()
    lower = text.lower()

    if len(text) < 3:
        return False

    if re.fullmatch(r"[\W_]+", lower):
        return False

    bad_patterns = [
        "omitted",
        "this message was edited",
        "this message was deleted",
        "voice call",
        "video call",
        "mídia oculta",
        "midia oculta",
        "mensagem apagada",
    ]

    if any(pattern in lower for pattern in bad_patterns):
        return False

    return True


def _select_sample_messages(messages: list[str]) -> dict[str, str]:
    """
    Seleciona mensagens reais de exemplo.
    Limita a mensagem longa para não explodir a tela.
    """
    valid_messages = [
        str(message).strip()
        for message in messages
        if _is_valid_sample_message(str(message))
    ]

    if not valid_messages:
        return {"short": "", "medium": "", "long": ""}

    sorted_by_length = sorted(valid_messages, key=lambda message: len(message))

    short = sorted_by_length[0]
    medium = sorted_by_length[len(sorted_by_length) // 2]
    long = sorted_by_length[-1]

    if len(long) > 400:
        long = long[:400] + "..."

    return {
        "short": short,
        "medium": medium,
        "long": long,
    }



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

    missing_columns = _REQUIRED_COLUMNS - set(messages_dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"O DataFrame não contém as colunas obrigatórias: {missing_columns}. "
            f"Colunas encontradas: {list(messages_dataframe.columns)}"
        )


    author_rows: pd.DataFrame = messages_dataframe[
        messages_dataframe["author"] == author_name
    ]

    if author_rows.empty:
        available = messages_dataframe["author"].unique().tolist()
        raise ValueError(
            f"Nenhuma mensagem encontrada para o autor '{author_name}'. "
            f"Autores disponíveis: {available}"
        )


    raw_texts: list[str] = [str(t) for t in author_rows["text"]]
    cleaned_texts: list[str] = [clean_text_for_stylometry(t) for t in raw_texts]


    all_features: list[dict[str, float | int]] = [
        extract_stylometric_features(t) for t in raw_texts
    ]

    message_count: int = len(all_features)

    def _avg(key: str) -> float:
        return sum(f[key] for f in all_features) / message_count

    def _total(key: str) -> int:
        return sum(f[key] for f in all_features)

    profile: dict[str, object] = {
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

        "top_words": _build_top_words(cleaned_texts),
        "top_bigrams": _build_top_bigrams(cleaned_texts),

        "sample_messages": _select_sample_messages(raw_texts),
    }

    return profile
