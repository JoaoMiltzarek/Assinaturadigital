"""Constrói perfis estilométricos agregados por autor."""

from __future__ import annotations

from collections import Counter
import re
import unicodedata

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

_NOISE_PROFILE_TOKENS = {
    "voice", "call", "message", "edited", "deleted", "omitted",
    "sticker", "audio", "image", "video", "gif", "document",
    "location", "contact", "card", "missed",
    "this", "was",
    "mídia", "midia", "oculta",
    "imagem", "áudio", "audio", "vídeo", "video",
    "figurinha", "omitida", "omitido",
}

_KNOWN_ABBREVIATIONS = {
    "vc": ["você", "voce"],
    "pq": ["porque", "por que"],
    "tbm": ["também", "tambem"],
    "hj": ["hoje"],
    "to": ["estou", "eu estou"],
    "tô": ["estou", "eu estou"],
    "ta": ["está", "esta"],
    "tá": ["está", "esta"],
    "pra": ["para"],
    "bora": ["vamos"],
}

_LAUGHTER_PATTERN = re.compile(
    r"\b(k{3,}|(ha){2,}|(rs){2,}|(he){2,})\b",
    re.IGNORECASE,
)

def _fold_token(text: str) -> str:
    text = str(text).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text


def _build_blocked_author_terms(messages_dataframe: pd.DataFrame) -> set[str]:
    blocked = set()

    if "author" not in messages_dataframe.columns:
        return blocked

    for author in messages_dataframe["author"].dropna().unique():
        parts = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", str(author).lower())

        for part in parts:
            folded = _fold_token(part)

            if len(folded) >= 2:
                blocked.add(folded)

    return blocked

def _normalize_profile_token(token: str, blocked_terms: set[str] | None = None) -> str:
    blocked_terms = blocked_terms or set()

    token = str(token).lower().strip()
    token = token.strip("[](){}.,;:!?\"'“”‘’<>")

    if not token:
        return ""

    folded = _fold_token(token)

    if folded in blocked_terms:
        return ""

    if re.fullmatch(r"\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}", token):
        return ""

    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", token):
        return ""

    if token.startswith("http") or token.startswith("www"):
        return ""

    if token in _NOISE_PROFILE_TOKENS or folded in _NOISE_PROFILE_TOKENS:
        return ""

    if token.isdigit():
        return ""

    if re.fullmatch(r"[\W_]+", token):
        return ""

    return token


def _tokenize_words_for_profile(
    text: str,
    blocked_terms: set[str] | None = None,
) -> list[str]:
    raw_words = str(text).lower().split()
    words = []

    for raw_word in raw_words:
        word = _normalize_profile_token(raw_word, blocked_terms)

        if len(word) >= _MIN_WORD_LENGTH and word not in _STOPWORDS:
            words.append(word)

    return words


def _tokenize_all_words(
    text: str,
    blocked_terms: set[str] | None = None,
) -> list[str]:
    raw_words = str(text).lower().split()
    words = []

    for raw_word in raw_words:
        word = _normalize_profile_token(raw_word, blocked_terms)

        if len(word) >= _MIN_WORD_LENGTH:
            words.append(word)

    return words


def _build_top_words(
    cleaned_texts: list[str],
    blocked_terms: set[str] | None = None,
) -> list[dict[str, object]]:
    counter: Counter[str] = Counter()
    for text in cleaned_texts:
        tokens = _tokenize_words_for_profile(text, blocked_terms)
        counter.update(tokens)

    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(_TOP_N)
    ]


def _build_top_bigrams(
    cleaned_texts: list[str],
    blocked_terms: set[str] | None = None,
) -> list[dict[str, object]]:
    counter: Counter[str] = Counter()
    for text in cleaned_texts:
        tokens = _tokenize_all_words(text, blocked_terms)

        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        counter.update(bigrams)

    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(_TOP_N)
    ]


def _is_valid_sample_message(message: str) -> bool:
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


def _detect_abbreviations(texts: list[str]) -> dict[str, int]:
    counter = Counter()

    for text in texts:
        tokens = re.findall(r"[A-Za-zÀ-ÿ0-9_]+", str(text).lower())

        for token in tokens:
            if token in _KNOWN_ABBREVIATIONS:
                counter[token] += 1

    return dict(counter)


def _detect_laughter_patterns(texts: list[str]) -> dict[str, int]:
    counter = Counter()

    for text in texts:
        matches = _LAUGHTER_PATTERN.findall(str(text).lower())

        for match in matches:
            laughter = match[0]
            if laughter:
                counter[laughter] += 1

    return dict(counter)


def build_author_stylometric_profile(
    messages_dataframe: pd.DataFrame,
    author_name: str,
) -> dict[str, object]:
    """Gera o perfil estilométrico de um autor."""
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

    blocked_terms = _build_blocked_author_terms(messages_dataframe)


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

        "top_words": _build_top_words(cleaned_texts, blocked_terms),
        "top_bigrams": _build_top_bigrams(cleaned_texts, blocked_terms),
        "detected_abbreviations": _detect_abbreviations(raw_texts),
        "laughter_patterns": _detect_laughter_patterns(raw_texts),
        "sample_messages": _select_sample_messages(raw_texts),
    }

    return profile
