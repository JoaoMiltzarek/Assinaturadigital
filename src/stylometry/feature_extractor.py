from __future__ import annotations

import re

import emoji

from src.processing.cleaner import clean_text_for_stylometry

_HASHTAG_PATTERN = re.compile(r"#\w+")
_MENTION_PATTERN = re.compile(r"@\w+")
_LAUGHTER_KKK_PATTERN = re.compile(r"\bk{3,}\b", re.IGNORECASE)
_LAUGHTER_HAHA_PATTERN = re.compile(r"\b(ha){2,}\b", re.IGNORECASE)
_LAUGHTER_RSRS_PATTERN = re.compile(r"\b(rs){2,}\b", re.IGNORECASE)
_LAUGHTER_HEHE_PATTERN = re.compile(r"\b(he){2,}\b", re.IGNORECASE)
_PUNCTUATION_CHARS = set(".,;:!?-–…")


def _count_emojis(text: str) -> int:
    return sum(1 for char in text if char in emoji.EMOJI_DATA)


def extract_stylometric_features(text: object) -> dict[str, float | int]:
    """Extrai features numéricas de uma mensagem."""
    raw: str = "" if text is None else str(text)
    cleaned: str = clean_text_for_stylometry(raw)

    words = cleaned.split() if cleaned else []
    total_words: int = len(words)
    safe_word_count = max(1, total_words)

    word_lengths = [len(w) for w in words]
    avg_word_length: float = sum(word_lengths) / safe_word_count

    num_unique_words: int = len(set(words))
    unique_ratio: float = num_unique_words / safe_word_count
    num_chars: int = len(cleaned)
    safe_char_count = max(1, num_chars)

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

    num_laughter_kkk: int = len(_LAUGHTER_KKK_PATTERN.findall(raw))
    num_laughter_haha: int = len(_LAUGHTER_HAHA_PATTERN.findall(raw))
    num_laughter_rsrs: int = len(_LAUGHTER_RSRS_PATTERN.findall(raw))
    num_laughter_hehe: int = len(_LAUGHTER_HEHE_PATTERN.findall(raw))
    total_laughter: int = (
        num_laughter_kkk + num_laughter_haha + num_laughter_rsrs + num_laughter_hehe
    )

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
