"""
src/stylometry/profile_builder.py

Construtor de perfil estilométrico agregado para um autor específico.

Responsabilidade única: dado um DataFrame com múltiplas mensagens e o nome
de um autor, calcular as médias e totais das features estilométricas e
retornar um dicionário de perfil numérico.

Esta é a Etapa 4A — o perfil mínimo. Etapas futuras adicionarão
top words, bigramas, exemplos de mensagens e resumo textual.
"""

from __future__ import annotations

import pandas as pd

from src.stylometry.feature_extractor import extract_stylometric_features

# Colunas obrigatórias que o DataFrame de entrada deve ter
_REQUIRED_COLUMNS = {"author", "text"}


def build_author_stylometric_profile(
    messages_dataframe: pd.DataFrame,
    author_name: str,
) -> dict[str, object]:
    """
    Constrói um perfil estilométrico agregado para um autor específico.

    Filtra as mensagens do autor no DataFrame, extrai as features de cada
    mensagem usando extract_stylometric_features e agrega os resultados
    em médias e totais.

    Args:
        messages_dataframe: DataFrame com pelo menos as colunas 'author' e 'text'.
        author_name: Nome exato do autor cujo perfil será construído.

    Returns:
        Dicionário com as seguintes chaves:
            author               : nome do autor
            message_count        : número de mensagens analisadas
            avg_words_per_message: média de palavras por mensagem
            avg_chars_per_message: média de caracteres por mensagem
            avg_word_length      : média do comprimento das palavras
            avg_unique_ratio     : média da proporção de palavras únicas
            avg_uppercase_ratio  : média da proporção de letras maiúsculas
            avg_punctuation_ratio: média da proporção de pontuação
            avg_emoji_ratio      : média da proporção de emojis por palavra
            total_emojis         : total de emojis em todas as mensagens
            total_laughter       : total de risadas (kkk, haha, rsrs, hehe)
            total_exclamation    : total de '!' em todas as mensagens
            total_question       : total de '?' em todas as mensagens

    Raises:
        ValueError: Se as colunas 'author' ou 'text' não existirem no DataFrame.
        ValueError: Se o autor não tiver nenhuma mensagem no DataFrame.
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
    author_messages: pd.DataFrame = messages_dataframe[
        messages_dataframe["author"] == author_name
    ]

    if author_messages.empty:
        autores_disponiveis = messages_dataframe["author"].unique().tolist()
        raise ValueError(
            f"Nenhuma mensagem encontrada para o autor '{author_name}'. "
            f"Autores disponíveis no DataFrame: {autores_disponiveis}"
        )

    # ------------------------------------------------------------------
    # Extração de features por mensagem
    # ------------------------------------------------------------------
    # Lista de dicionários, um por mensagem
    all_features: list[dict[str, float | int]] = [
        extract_stylometric_features(text)
        for text in author_messages["text"]
    ]

    message_count: int = len(all_features)

    # Função auxiliar interna para calcular a média de uma feature
    def _avg(feature_name: str) -> float:
        values = [f[feature_name] for f in all_features]
        return sum(values) / message_count

    # Função auxiliar interna para somar uma feature
    def _total(feature_name: str) -> int:
        return sum(f[feature_name] for f in all_features)

    # ------------------------------------------------------------------
    # Agregação do perfil
    # ------------------------------------------------------------------
    profile: dict[str, object] = {
        "author": author_name,
        "message_count": message_count,

        # Médias por mensagem
        "avg_words_per_message": _avg("num_words"),
        "avg_chars_per_message": _avg("num_chars"),
        "avg_word_length": _avg("avg_word_length"),
        "avg_unique_ratio": _avg("unique_ratio"),
        "avg_uppercase_ratio": _avg("uppercase_ratio"),
        "avg_punctuation_ratio": _avg("punctuation_ratio"),
        "avg_emoji_ratio": _avg("emoji_ratio"),

        # Totais acumulados
        "total_emojis": _total("num_emojis"),
        "total_laughter": _total("total_laughter"),
        "total_exclamation": _total("num_exclamation"),
        "total_question": _total("num_question"),
    }

    return profile
