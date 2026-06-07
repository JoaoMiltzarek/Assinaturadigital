from __future__ import annotations

import re


def _get_float(profile: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(profile.get(key, default))
    except (TypeError, ValueError):
        return default

def _get_int(profile: dict, key: str, default: int = 0) -> int:
    try:
        return int(profile.get(key, default))
    except (TypeError, ValueError):
        return default


def _extract_term(item) -> str:
    if isinstance(item, dict):
        for key in ["term", "word", "bigram", "text", "value"]:
            if key in item:
                return str(item[key]).strip()

    if isinstance(item, (list, tuple)) and len(item) > 0:
        return str(item[0]).strip()

    return str(item).strip()


def _get_profile_terms(profile: dict) -> set[str]:
    terms = set()

    for key in ["top_words", "top_bigrams"]:
        for item in profile.get(key, []) or []:
            term = _extract_term(item).lower()
            if term:
                terms.add(term)

    return terms


def _author_uses(profile: dict, expression: str) -> bool:
    expression = expression.lower().strip()

    detected_abbreviations = profile.get("detected_abbreviations", {}) or {}

    if expression in detected_abbreviations and detected_abbreviations[expression] > 0:
        return True

    terms = _get_profile_terms(profile)

    return expression in terms


def _apply_only_author_abbreviations(text: str, profile: dict) -> str:
    """
    Só aplica abreviações que aparecem no perfil do autor.
    Isso evita inventar 'hj' se o autor não usa 'hj'.
    """
    replacements = []

    if _author_uses(profile, "vc"):
        replacements.append((r"\bvocê\b", "vc"))
        replacements.append((r"\bvoce\b", "vc"))

    if _author_uses(profile, "pq"):
        replacements.append((r"\bpor que\b", "pq"))
        replacements.append((r"\bporque\b", "pq"))

    if _author_uses(profile, "tbm"):
        replacements.append((r"\btambém\b", "tbm"))
        replacements.append((r"\btambem\b", "tbm"))

    if _author_uses(profile, "hj"):
        replacements.append((r"\bhoje\b", "hj"))

    if _author_uses(profile, "to") or _author_uses(profile, "tô"):
        replacements.append((r"\bestou\b", "to"))
        replacements.append((r"\beu estou\b", "to"))

    if _author_uses(profile, "ta") or _author_uses(profile, "tá"):
        replacements.append((r"\bestá\b", "ta"))
        replacements.append((r"\besta\b", "ta"))

    if _author_uses(profile, "pra"):
        replacements.append((r"\bpara\b", "pra"))

    if _author_uses(profile, "bora"):
        replacements.append((r"\bvamos\b", "bora"))

    result = text

    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def _author_laughter(profile: dict) -> str:
    laughter_patterns = profile.get("laughter_patterns", {}) or {}

    if laughter_patterns:
        return max(laughter_patterns, key=laughter_patterns.get)

    terms = _get_profile_terms(profile)

    for laughter in ["kkkkkk", "kkkkk", "kkkk", "kkk", "haha", "rsrs", "hehe"]:
        if laughter in terms:
            return laughter

    return ""


def _choose_final_punctuation(message: str, profile: dict, intensity: int) -> str:
    message_count = max(1, _get_int(profile, "message_count", 1))

    total_exclamation = _get_float(profile, "total_exclamation", 0.0)
    total_question = _get_float(profile, "total_question", 0.0)

    exclamation_rate = total_exclamation / message_count
    question_rate = total_question / message_count

    if "?" in message:
        return "?"

    if exclamation_rate > 0.08 and intensity >= 2:
        return "!"

    if question_rate > 0.12 and intensity == 3:
        return "?"

    return ""


def rewrite_message_with_profile(
    message: str,
    profile: dict,
    intensity: int = 2,
) -> str:
    """
    Recria uma mensagem usando regras baseadas no perfil real.

    Não inventa abreviações que o autor não usa.
    Não corta palavra no meio.
    Não reduz a frase para uma palavra só.
    """
    original = str(message).strip()

    if not original:
        return ""

    intensity = max(1, min(intensity, 3))

    text = original.strip()

    # Se o autor escreve majoritariamente em minúsculo, baixa a frase.
    avg_uppercase_ratio = _get_float(profile, "avg_uppercase_ratio", 0.0)

    if avg_uppercase_ratio < 0.04:
        text = text.lower()

    # Aplica apenas abreviações reais do autor.
    text = _apply_only_author_abbreviations(text, profile)

    # Remove pontuação final para reconstruir.
    text = re.sub(r"[.!?…]+$", "", text).strip()

    # Adiciona pontuação de acordo com o perfil.
    punctuation = _choose_final_punctuation(original, profile, intensity)

    if punctuation:
        text = text + punctuation

    # Adiciona risada apenas se ela aparece no perfil.
    laughter = _author_laughter(profile)
    total_laughter = _get_float(profile, "total_laughter", 0.0)
    message_count = max(1, _get_int(profile, "message_count", 1))
    laughter_rate = total_laughter / message_count

    if laughter and laughter_rate >= 0.03 and intensity >= 3:
        if laughter not in text.lower():
            text = text + " " + laughter

    return text.strip()


def explain_rewrite_rules(profile: dict) -> list[str]:
    message_count = _get_int(profile, "message_count", 0)
    avg_words = _get_float(profile, "avg_words_per_message", 0.0)
    avg_chars = _get_float(profile, "avg_chars_per_message", 0.0)

    terms = _get_profile_terms(profile)
    detected_abbreviations = profile.get("detected_abbreviations", {}) or {}
    laughter_patterns = profile.get("laughter_patterns", {}) or {}

    rules = [
        f"Foram analisadas {message_count} mensagens do autor.",
        f"O autor escreve em média {avg_words:.1f} palavras por mensagem.",
        f"O tamanho médio das mensagens é de {avg_chars:.1f} caracteres.",
        "A recriação usa apenas abreviações encontradas no perfil real do autor.",
    ]

    used_abbreviations = []

    for abbreviation in ["vc", "pq", "tbm", "hj", "to", "tô", "ta", "tá", "pra", "bora"]:
        if abbreviation in detected_abbreviations or abbreviation in terms:
            used_abbreviations.append(abbreviation)

    if used_abbreviations:
        rules.append(
            "Abreviações detectadas no perfil: " + ", ".join(used_abbreviations) + "."
        )
    else:
        rules.append("Nenhuma abreviação forte foi detectada no perfil, então o sistema evita inventar abreviações.")

    return rules