"""
src/loaders/generic_text_loader.py

Loader simples para transformar um bloco de texto puro em mensagens
normalizadas (uma linha = uma mensagem).
"""

from __future__ import annotations

import pandas as pd

from src.processing.normalizer import normalize_messages_dataframe

def load_generic_text_messages(
    text_content: str,
    author_name: str = "unknown_author",
    source_name: str = "text",
) -> pd.DataFrame:
    """
    Transforma texto puro em mensagens. Cada linha não vazia vira uma mensagem.
    
    Args:
        text_content: String contendo o texto completo.
        author_name: Nome a ser atribuído a todas as mensagens.
        source_name: Nome da fonte.
        
    Returns:
        DataFrame padronizado com as colunas padrão.
    """
    # Divide em linhas e remove linhas vazias
    lines = [line.strip() for line in text_content.splitlines() if line.strip()]
    
    # Cria o DataFrame bruto
    raw_df = pd.DataFrame({
        "author": [author_name] * len(lines),
        "text": lines,
    })
    
    # Se não tiver mensagens, retorna o df bruto normalizado que ficará vazio
    if raw_df.empty:
        raw_df = pd.DataFrame(columns=["author", "text"])
        
    return normalize_messages_dataframe(
        raw_dataframe=raw_df,
        author_column="author",
        text_column="text",
        source_name=source_name,
    )
