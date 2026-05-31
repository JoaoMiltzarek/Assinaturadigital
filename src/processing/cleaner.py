"""
src/processing/cleaner.py

Funções de limpeza textual reutilizáveis para o pipeline do AssinaturaDigital.

Responsabilidade única: receber qualquer string bruta e devolver uma string
limpa, pronta para extração de features estilométricas.

NÃO remove acentos, emojis, maiúsculas ou pontuação comum —
essas características são relevantes para a análise de estilo.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Padrões regex compilados como constantes privadas (compilar uma vez,
# usar muitas vezes — mais eficiente do que re.sub(r"...", ...))
# ---------------------------------------------------------------------------

# URLs: http://..., https://..., www....
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Menções: @usuario
_MENTION_PATTERN = re.compile(r"@\w+")

# Apenas o símbolo '#' de hashtags — mantém a palavra que vem depois
_HASHTAG_SYMBOL_PATTERN = re.compile(r"#(\w+)")

# Dois ou mais espaços em sequência
_MULTIPLE_SPACES_PATTERN = re.compile(r" {2,}")


def clean_text_for_stylometry(text: object) -> str:
    """
    Limpa um texto bruto para uso na extração de features estilométricas.

    Etapas aplicadas (nessa ordem):
    1. Converte a entrada para string de forma segura (None → "").
    2. Remove URLs (http, https, www).
    3. Remove menções (@usuario).
    4. Remove o símbolo '#' de hashtags, mas mantém a palavra.
    5. Normaliza espaços múltiplos em um único espaço.
    6. Remove espaços no início e no final (strip).

    O que NÃO é removido: emojis, acentos, maiúsculas, pontuação comum.

    Args:
        text: Qualquer valor — string, número, None, etc.

    Returns:
        String limpa, ou string vazia se a entrada for None ou vazia.

    Exemplos:
        >>> clean_text_for_stylometry("Oi @joao veja https://site.com #Teste agora")
        'Oi veja Teste agora'
        >>> clean_text_for_stylometry(None)
        ''
    """
    # Passo 1: conversão segura para string
    if text is None:
        return ""
    cleaned: str = str(text)

    # Passo 2: remove URLs
    cleaned = _URL_PATTERN.sub("", cleaned)

    # Passo 3: remove menções (@usuario)
    cleaned = _MENTION_PATTERN.sub("", cleaned)

    # Passo 4: remove o '#' mas mantém a palavra da hashtag
    # Ex: "#Teste" → "Teste"
    cleaned = _HASHTAG_SYMBOL_PATTERN.sub(r"\1", cleaned)

    # Passo 5: normaliza espaços múltiplos
    cleaned = _MULTIPLE_SPACES_PATTERN.sub(" ", cleaned)

    # Passo 6: strip de bordas
    cleaned = cleaned.strip()

    return cleaned
