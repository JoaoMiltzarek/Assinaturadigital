"""
src/stylometry/feature_extractor.py

Extração de características estilométricas de uma única mensagem de texto.

Responsabilidade única: dado um texto bruto, retornar um dicionário de
features numéricas que representam o estilo de escrita do autor.

Usa clean_text_for_stylometry para a limpeza antes de calcular
palavras, caracteres e vocabulário — preservando emojis, acentos e
maiúsculas no texto original para features que dependem deles.
"""

from __future__ import annotations

import re

import emoji

from src.processing.cleaner import clean_text_for_stylometry

# ---------------------------------------------------------------------------
# Padrões compilados — contados no texto ORIGINAL (antes da limpeza)
# ---------------------------------------------------------------------------

# Hashtags: #palavra
_HASHTAG_PATTERN = re.compile(r"#\w+")

# Menções: @usuario
_MENTION_PATTERN = re.compile(r"@\w+")

# Risadas — padrão case-insensitive, com repetição de letras
# kkk, kkkk, kkkkk, ...  (mínimo 3 k's)
_LAUGHTER_KKK_PATTERN = re.compile(r"\bk{3,}\b", re.IGNORECASE)

# haha, hahaha, ... (mínimo "haha")
_LAUGHTER_HAHA_PATTERN = re.compile(r"\b(ha){2,}\b", re.IGNORECASE)

# rsrs, rsrsrs, ...
_LAUGHTER_RSRS_PATTERN = re.compile(r"\b(rs){2,}\b", re.IGNORECASE)

# hehe, hehehe, ...
_LAUGHTER_HEHE_PATTERN = re.compile(r"\b(he){2,}\b", re.IGNORECASE)

# Pontuação relevante para estilo
_PUNCTUATION_CHARS = set(".,;:!?-–…")


def _count_emojis(text: str) -> int:
    """Conta o número de emojis em um texto usando a biblioteca emoji."""
    return sum(1 for char in text if char in emoji.EMOJI_DATA)


def extract_stylometric_features(text: object) -> dict[str, float | int]:
    """
    Extrai características estilométricas de uma única mensagem.

    Recebe qualquer entrada (string, número, None) e retorna um dicionário
    com features numéricas que descrevem o estilo de escrita.

    Features retornadas:
        avg_word_length   : média do comprimento das palavras (texto limpo)
        num_words         : número total de palavras (texto limpo)
        num_unique_words  : número de palavras únicas (texto limpo)
        unique_ratio      : proporção de palavras únicas / total
        num_chars         : número de caracteres (texto limpo, sem espaços extra)
        num_punctuation   : contagem de pontuação no texto original
        uppercase_ratio   : proporção de letras maiúsculas / total de letras
        num_emojis        : número de emojis (texto original)
        num_hashtags      : número de hashtags (texto original)
        num_mentions      : número de menções @usuario (texto original)
        num_exclamation   : número de '!' (texto original)
        num_question      : número de '?' (texto original)
        num_laughter_kkk  : ocorrências de "kkk..." (texto original)
        num_laughter_haha : ocorrências de "haha..." (texto original)
        num_laughter_rsrs : ocorrências de "rsrs..." (texto original)
        num_laughter_hehe : ocorrências de "hehe..." (texto original)
        total_laughter    : soma de todas as formas de risada
        punctuation_ratio : pontuação / num_chars (evita divisão por zero)
        emoji_ratio       : emojis / num_words (evita divisão por zero)

    Args:
        text: Qualquer valor — será convertido para string de forma segura.

    Returns:
        Dicionário com todas as features numéricas listadas acima.
    """
    # Conversão segura — None vira string vazia
    raw: str = "" if text is None else str(text)

    # Texto limpo: sem URLs, menções, símbolo de hashtag e espaços extras.
    # Usado para calcular palavras e vocabulário.
    cleaned: str = clean_text_for_stylometry(raw)

    # -----------------------------------------------------------------
    # Features baseadas no texto LIMPO
    # -----------------------------------------------------------------
    words = cleaned.split() if cleaned else []
    total_words: int = len(words)

    # Evita divisão por zero usando max(1, ...)
    safe_word_count = max(1, total_words)

    word_lengths = [len(w) for w in words]
    avg_word_length: float = sum(word_lengths) / safe_word_count

    num_unique_words: int = len(set(words))
    unique_ratio: float = num_unique_words / safe_word_count
    num_chars: int = len(cleaned)
    safe_char_count = max(1, num_chars)

    # -----------------------------------------------------------------
    # Features baseadas no texto ORIGINAL (preserva hashtags, @, emojis)
    # -----------------------------------------------------------------
    letters = [c for c in raw if c.isalpha()]
    uppercase_ratio: float = (
        sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    )

    num_punctuation: int = sum(1 for c in raw if c in _PUNCTUATION_CHARS)
    num_exclamation: int = raw.count("!")
    num_question: int = raw.count("?")

    num_emojis: int = _count_emojis(raw)
    num_hashtags: int = len(_HASHTAG_PATTERN.findall(raw))
    num_mentions: int = len(_MENTION_PATTERN.findall(raw))

    # -----------------------------------------------------------------
    # Risadas
    # -----------------------------------------------------------------
    num_laughter_kkk: int = len(_LAUGHTER_KKK_PATTERN.findall(raw))
    num_laughter_haha: int = len(_LAUGHTER_HAHA_PATTERN.findall(raw))
    num_laughter_rsrs: int = len(_LAUGHTER_RSRS_PATTERN.findall(raw))
    num_laughter_hehe: int = len(_LAUGHTER_HEHE_PATTERN.findall(raw))
    total_laughter: int = (
        num_laughter_kkk + num_laughter_haha + num_laughter_rsrs + num_laughter_hehe
    )

    # -----------------------------------------------------------------
    # Ratios derivados
    # -----------------------------------------------------------------
    punctuation_ratio: float = num_punctuation / safe_char_count
    emoji_ratio: float = num_emojis / safe_word_count

    return {
        "avg_word_length": avg_word_length,
        "num_words": total_words,
        "num_unique_words": num_unique_words,
        "unique_ratio": unique_ratio,
        "num_chars": num_chars,
        "num_punctuation": num_punctuation,
        "uppercase_ratio": uppercase_ratio,
        "num_emojis": num_emojis,
        "num_hashtags": num_hashtags,
        "num_mentions": num_mentions,
        "num_exclamation": num_exclamation,
        "num_question": num_question,
        "num_laughter_kkk": num_laughter_kkk,
        "num_laughter_haha": num_laughter_haha,
        "num_laughter_rsrs": num_laughter_rsrs,
        "num_laughter_hehe": num_laughter_hehe,
        "total_laughter": total_laughter,
        "punctuation_ratio": punctuation_ratio,
        "emoji_ratio": emoji_ratio,
    }
