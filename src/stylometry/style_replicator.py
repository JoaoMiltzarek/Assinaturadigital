from __future__ import annotations

import re
from collections import Counter
from typing import Any


# ============================================================
# style_replicator.py
#
# Recriador simples de estilo por regras.
#
# Objetivo acadêmico:
# - não usar LLM;
# - não inventar padrões;
# - aplicar apenas sinais observados no perfil estilométrico;
# - preservar o sentido da mensagem original.
# ============================================================


_ABBREVIATION_RULES: dict[str, list[tuple[str, str]]] = {
    "vc": [
        (r"\bvocê\b", "vc"),
        (r"\bvoce\b", "vc"),
    ],
    "cê": [
        (r"\bvocê\b", "cê"),
        (r"\bvoce\b", "cê"),
    ],
    "ce": [
        (r"\bvocê\b", "ce"),
        (r"\bvoce\b", "ce"),
    ],
    "pq": [
        (r"\bpor que\b", "pq"),
        (r"\bporque\b", "pq"),
    ],
    "tbm": [
        (r"\btambém\b", "tbm"),
        (r"\btambem\b", "tbm"),
    ],
    "hj": [
        (r"\bhoje\b", "hj"),
    ],
    "tô": [
        (r"\beu estou\b", "tô"),
        (r"\bestou\b", "tô"),
    ],
    "to": [
        (r"\beu estou\b", "to"),
        (r"\bestou\b", "to"),
    ],
    "tá": [
        (r"\bestá\b", "tá"),
        (r"\besta\b", "tá"),
    ],
    "ta": [
        (r"\bestá\b", "ta"),
        (r"\besta\b", "ta"),
    ],
    "pra": [
        (r"\bpara\b", "pra"),
    ],
    "bora": [
        (r"\bvamos\b", "bora"),
    ],
    "mano": [],
    "cara": [],
    "tipo": [],
}

_ABBREVIATION_PRIORITY = [
    "vc", "cê", "ce", "pq", "tbm", "hj", "tô", "to", "tá", "ta", "pra", "bora",
]

_LAUGHTER_PATTERN = re.compile(
    r"\b(k{3,}|(?:ha){2,}|(?:rs){2,}|(?:he){2,})\b",
    flags=re.IGNORECASE,
)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+"
)


# ============================================================
# Conversões seguras
# ============================================================


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


# ============================================================
# Leitura robusta do perfil
# ============================================================


def _extract_term(item: Any) -> str:
    """
    Extrai termos de estruturas diferentes.

    Compatível com:
    - {"term": "kkk", "count": 3}
    - {"word": "kkk", "count": 3}
    - {"bigram": "muito bom", "count": 2}
    - ("kkk", 3)
    - "kkk"
    """
    if isinstance(item, dict):
        for key in ["term", "word", "bigram", "text", "value", "token"]:
            if key in item:
                return str(item[key]).strip()

    if isinstance(item, (list, tuple)) and len(item) > 0:
        return str(item[0]).strip()

    return str(item).strip()


def _get_profile_terms(profile: dict) -> set[str]:
    """
    Junta palavras e bigramas frequentes em um conjunto simples.
    """
    terms: set[str] = set()

    for key in ["top_words", "top_bigrams"]:
        for item in profile.get(key, []) or []:
            term = _extract_term(item).lower().strip()
            if term:
                terms.add(term)

    return terms


def _get_sample_messages(profile: dict) -> list[str]:
    """
    Retorna exemplos reais do perfil, quando existirem.
    """
    samples = profile.get("sample_messages", {}) or {}

    if isinstance(samples, dict):
        return [str(value).strip() for value in samples.values() if str(value).strip()]

    if isinstance(samples, list):
        return [str(value).strip() for value in samples if str(value).strip()]

    return []


def _get_profile_text_blob(profile: dict) -> str:
    """
    Cria um texto auxiliar com termos e exemplos reais.

    Isso permite detectar padrões mesmo quando o profile_builder ainda não
    possui chaves explícitas como detected_abbreviations ou laughter_patterns.
    """
    terms = sorted(_get_profile_terms(profile))
    samples = _get_sample_messages(profile)
    return " ".join(terms + samples).lower()


# ============================================================
# Detecção de padrões reais do autor
# ============================================================


def _author_uses(profile: dict, expression: str) -> bool:
    """
    Verifica se uma expressão aparece no perfil real.

    Ordem:
    1. detected_abbreviations, se existir;
    2. top_words/top_bigrams;
    3. sample_messages.

    Isso evita inventar abreviações que não aparecem no autor.
    """
    expression = expression.lower().strip()

    detected_abbreviations = profile.get("detected_abbreviations", {}) or {}

    if expression in detected_abbreviations and detected_abbreviations[expression] > 0:
        return True

    terms = _get_profile_terms(profile)

    if expression in terms:
        return True

    blob = _get_profile_text_blob(profile)
    return re.search(rf"\b{re.escape(expression)}\b", blob, flags=re.IGNORECASE) is not None


def _detect_author_abbreviations(profile: dict) -> list[str]:
    """
    Lista abreviações realmente detectadas no perfil.
    """
    detected: list[str] = []

    for abbreviation in _ABBREVIATION_PRIORITY:
        if _author_uses(profile, abbreviation):
            detected.append(abbreviation)

    return detected


def _detect_laughter_counter(profile: dict) -> Counter[str]:
    """
    Detecta risadas no perfil.

    Usa laughter_patterns quando existe, mas também consegue funcionar
    só com top_words e sample_messages.
    """
    counter: Counter[str] = Counter()

    laughter_patterns = profile.get("laughter_patterns", {}) or {}

    if isinstance(laughter_patterns, dict):
        for laughter, count in laughter_patterns.items():
            if str(laughter).strip():
                try:
                    counter[str(laughter).lower().strip()] += int(count)
                except (TypeError, ValueError):
                    counter[str(laughter).lower().strip()] += 1

    blob = _get_profile_text_blob(profile)

    for match in _LAUGHTER_PATTERN.findall(blob):
        if match:
            counter[str(match).lower()] += 1

    return counter


def _author_laughter(profile: dict) -> str:
    counter = _detect_laughter_counter(profile)

    if not counter:
        return ""

    return counter.most_common(1)[0][0]


def _detect_common_emojis(profile: dict) -> list[str]:
    """
    Detecta emojis reais nos exemplos do perfil.

    Se não houver emojis nos exemplos, não inventa emoji.
    """
    blob = " ".join(_get_sample_messages(profile))
    emojis = _EMOJI_PATTERN.findall(blob)

    if not emojis:
        return []

    counter: Counter[str] = Counter()

    for group in emojis:
        for char in group:
            counter[char] += 1

    return [emoji for emoji, _ in counter.most_common(3)]


# ============================================================
# Capitalização
# ============================================================


def _first_alpha_char(text: str) -> str:
    for char in str(text):
        if char.isalpha():
            return char
    return ""


def _capitalize_first_letter(text: str) -> str:
    if not text:
        return text

    chars = list(text)

    for index, char in enumerate(chars):
        if char.isalpha():
            chars[index] = char.upper()
            break

    return "".join(chars)


def _sample_initial_capital_ratio(profile: dict) -> float | None:
    samples = _get_sample_messages(profile)

    if not samples:
        return None

    valid_initials = []

    for sample in samples:
        first = _first_alpha_char(sample)
        if first:
            valid_initials.append(first.isupper())

    if not valid_initials:
        return None

    return sum(valid_initials) / len(valid_initials)


def _should_lowercase(profile: dict) -> bool:
    """
    Decide se o autor tende a escrever tudo/minimamente em minúsculo.

    Usa samples primeiro, porque avg_uppercase_ratio pode ser enganoso
    quando há poucas mensagens ou nomes próprios.
    """
    capital_ratio = _sample_initial_capital_ratio(profile)
    avg_uppercase_ratio = _get_float(profile, "avg_uppercase_ratio", 0.0)

    if capital_ratio is not None:
        return capital_ratio <= 0.25 and avg_uppercase_ratio < 0.06

    return avg_uppercase_ratio < 0.035


def _should_capitalize_initial(profile: dict) -> bool:
    capital_ratio = _sample_initial_capital_ratio(profile)
    avg_uppercase_ratio = _get_float(profile, "avg_uppercase_ratio", 0.0)

    if capital_ratio is not None:
        return capital_ratio >= 0.5

    return avg_uppercase_ratio >= 0.06


def _apply_capitalization_style(text: str, profile: dict) -> str:
    """
    Aplica capitalização sem exagerar.

    - Se o autor escreve quase tudo minúsculo, deixa minúsculo.
    - Se o autor começa frases com maiúscula, capitaliza a primeira letra.
    - Nunca inventa CAIXA ALTA.
    """
    if _should_lowercase(profile):
        return text.lower()

    if _should_capitalize_initial(profile):
        return _capitalize_first_letter(text)

    return text


# ============================================================
# Abreviações e substituições
# ============================================================


def _apply_only_author_abbreviations(text: str, profile: dict, intensity: int) -> str:
    """
    Aplica apenas abreviações encontradas no perfil real.

    Intensidade:
    - 1: não aplica abreviações, só estilo superficial;
    - 2 e 3: aplica abreviações detectadas.
    """
    if intensity <= 1:
        return text

    detected_abbreviations = _detect_author_abbreviations(profile)
    result = text

    for abbreviation in detected_abbreviations:
        rules = _ABBREVIATION_RULES.get(abbreviation, [])

        for pattern, replacement in rules:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


# ============================================================
# Pontuação
# ============================================================


def _author_uses_ellipsis(profile: dict) -> bool:
    samples = " ".join(_get_sample_messages(profile))
    return "..." in samples or "…" in samples


def _choose_final_punctuation(original_message: str, profile: dict, intensity: int) -> str:
    """
    Escolhe pontuação final com base no perfil e na mensagem original.

    Preserva pergunta quando a mensagem original é pergunta.
    Só adiciona exclamação/reticências se houver sinal no perfil.
    """
    original_message = str(original_message).strip()
    message_count = max(1, _get_int(profile, "message_count", 1))

    total_exclamation = _get_float(profile, "total_exclamation", 0.0)
    total_question = _get_float(profile, "total_question", 0.0)

    exclamation_rate = total_exclamation / message_count
    question_rate = total_question / message_count

    if original_message.endswith("?"):
        return "?"

    if original_message.endswith("!") and exclamation_rate > 0:
        return "!"

    if original_message.endswith(("...", "…")) and _author_uses_ellipsis(profile):
        return "..."

    if intensity >= 3 and _author_uses_ellipsis(profile):
        return "..."

    if exclamation_rate >= 0.15 and intensity >= 2:
        return "!"

    if question_rate >= 0.25 and intensity >= 3:
        return "?"

    return ""


def _strip_final_punctuation(text: str) -> str:
    return re.sub(r"[.!?…]+$", "", text).strip()


# ============================================================
# Adornos reais: risada e emoji
# ============================================================


def _should_add_laughter(profile: dict, intensity: int) -> bool:
    if intensity < 3:
        return False

    laughter = _author_laughter(profile)
    if not laughter:
        return False

    total_laughter = _get_float(profile, "total_laughter", 0.0)
    message_count = max(1, _get_int(profile, "message_count", 1))
    laughter_rate = total_laughter / message_count

    # Se o total_laughter não veio preenchido, mas a risada apareceu nos samples,
    # ainda permitimos em intensidade 3.
    return laughter_rate >= 0.03 or bool(_detect_laughter_counter(profile))


def _add_laughter_if_allowed(text: str, profile: dict, intensity: int) -> str:
    if not _should_add_laughter(profile, intensity):
        return text

    laughter = _author_laughter(profile)

    if not laughter:
        return text

    if laughter.lower() in text.lower():
        return text

    return f"{text} {laughter}"


def _add_emoji_if_allowed(text: str, profile: dict, intensity: int) -> str:
    """
    Adiciona emoji apenas quando ele aparece nos exemplos reais.
    """
    if intensity < 3:
        return text

    total_emojis = _get_float(profile, "total_emojis", 0.0)
    message_count = max(1, _get_int(profile, "message_count", 1))
    emoji_rate = total_emojis / message_count

    if emoji_rate < 0.05:
        return text

    emojis = _detect_common_emojis(profile)

    if not emojis:
        return text

    emoji = emojis[0]

    if emoji in text:
        return text

    return f"{text} {emoji}"


# ============================================================
# Funções públicas usadas pelo app.py
# ============================================================


def rewrite_message_with_profile(
    message: str,
    profile: dict,
    intensity: int = 2,
) -> str:
    """
    Recria uma mensagem usando regras baseadas no perfil real.

    O que esta função faz:
    - preserva o sentido da mensagem;
    - aplica capitalização/minúsculas conforme o perfil;
    - aplica abreviações somente se aparecem no perfil;
    - aplica pontuação, risadas e emojis somente se há evidência no perfil;
    - não usa LLM;
    - não inventa vocabulário novo.
    """
    original = "" if message is None else str(message).strip()

    if not original:
        return ""

    intensity = max(1, min(intensity, 3))

    text = original

    # Remove URLs para não estilizar lixo técnico.
    text = _URL_PATTERN.sub("", text).strip()

    # Aplica abreviações reais do autor.
    text = _apply_only_author_abbreviations(text, profile, intensity)

    # Reconstrói pontuação final de forma controlada.
    text = _strip_final_punctuation(text)
    punctuation = _choose_final_punctuation(original, profile, intensity)

    if punctuation:
        text = f"{text}{punctuation}"

    # Aplica estilo de maiúscula/minúscula no final, para corrigir substituições.
    text = _apply_capitalization_style(text, profile)

    # Adiciona risada e emoji só em intensidade alta e com evidência real.
    text = _add_laughter_if_allowed(text, profile, intensity)
    text = _add_emoji_if_allowed(text, profile, intensity)

    # Limpeza final de espaços duplicados.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def explain_rewrite_rules(profile: dict) -> list[str]:
    """
    Explica as regras aplicadas/possíveis de forma apresentável no Streamlit.
    """
    message_count = _get_int(profile, "message_count", 0)
    avg_words = _get_float(profile, "avg_words_per_message", 0.0)
    avg_chars = _get_float(profile, "avg_chars_per_message", 0.0)
    avg_uppercase_ratio = _get_float(profile, "avg_uppercase_ratio", 0.0)

    detected_abbreviations = _detect_author_abbreviations(profile)
    laughter = _author_laughter(profile)
    emojis = _detect_common_emojis(profile)

    rules = [
    f"Foram analisadas {message_count} mensagens do autor.",
    f"O autor escreve em média {avg_words:.1f} palavras por mensagem.",
    f"O tamanho médio das mensagens é de {avg_chars:.1f} caracteres.",
]

    capital_ratio = _sample_initial_capital_ratio(profile)

    if _should_lowercase(profile):
        rules.append(
            "O autor tende a usar poucas maiúsculas, então o sistema pode deixar a mensagem em minúsculas."
        )
    elif _should_capitalize_initial(profile):
        rules.append(
            "O autor costuma iniciar mensagens com letra maiúscula, então o sistema capitaliza a primeira letra."
        )
    else:
        rules.append(
            "Não há padrão forte de maiúsculas/minúsculas, então o sistema evita alterar demais a capitalização."
        )

    if capital_ratio is not None:
        rules.append(
            f"Nos exemplos do perfil, {capital_ratio * 100:.0f}% das mensagens começam com letra maiúscula."
        )
    else:
        rules.append(
            f"A proporção média de letras maiúsculas no perfil é {avg_uppercase_ratio:.3f}."
        )

    if detected_abbreviations:
        rules.append(
            "Abreviações detectadas no perfil: " + ", ".join(detected_abbreviations) + "."
        )
    else:
        rules.append(
            "Nenhuma abreviação forte foi detectada; por isso o sistema evita inventar 'vc', 'hj', 'pq' etc."
        )

    if laughter:
        rules.append(f"Risada detectada no perfil: {laughter}.")
    else:
        rules.append(
            "Nenhuma risada recorrente foi detectada; por isso o sistema evita inventar risadas."
        )

    if emojis:
        rules.append(
            "Emojis detectados nos exemplos reais: " + " ".join(emojis) + "."
        )
    else:
        rules.append(
            "Nenhum emoji recorrente foi encontrado nos exemplos; por isso o sistema evita inventar emojis."
        )

    if _author_uses_ellipsis(profile):
        rules.append("O autor usa reticências nos exemplos, então o sistema pode usar '...' em intensidade alta.")

    return rules