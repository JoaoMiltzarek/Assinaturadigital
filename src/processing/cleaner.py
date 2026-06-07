from __future__ import annotations

import re

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_SYMBOL_PATTERN = re.compile(r"#(\w+)")
_MULTIPLE_SPACES_PATTERN = re.compile(r" {2,}")


def clean_text_for_stylometry(text: object) -> str:
    """Limpa texto sem remover sinais úteis de estilo."""
    if text is None:
        return ""
    cleaned: str = str(text)

    cleaned = _URL_PATTERN.sub("", cleaned)
    cleaned = _MENTION_PATTERN.sub("", cleaned)
    cleaned = _HASHTAG_SYMBOL_PATTERN.sub(r"\1", cleaned)
    cleaned = _MULTIPLE_SPACES_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.strip()

    return cleaned
