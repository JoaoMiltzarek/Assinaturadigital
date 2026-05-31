"""
src/loaders/csv_loader.py

Loader simples para ler mensagens a partir de arquivos CSV e transformá-las
no formato interno padrão usando o normalizer.
"""

from __future__ import annotations

from typing import Any
import pandas as pd

from src.processing.normalizer import normalize_messages_dataframe

def load_csv_messages(
    csv_path_or_buffer: Any,
    author_column: str,
    text_column: str,
    datetime_column: str | None = None,
    metadata_column: str | None = None,
    source_name: str = "csv",
) -> pd.DataFrame:
    """
    Lê um arquivo CSV e retorna as mensagens normalizadas.
    
    Args:
        csv_path_or_buffer: Caminho do arquivo ou buffer de string (ex: StringIO).
        author_column: Nome da coluna no CSV que indica o autor.
        text_column: Nome da coluna no CSV que contém o texto da mensagem.
        datetime_column: Nome da coluna opcional de data/hora.
        metadata_column: Nome da coluna opcional para metadados.
        source_name: Nome da fonte constante.
        
    Returns:
        DataFrame padronizado com as colunas: 
        author, text, datetime, source, metadata.
    """
    raw_df = pd.read_csv(csv_path_or_buffer)
    
    return normalize_messages_dataframe(
        raw_dataframe=raw_df,
        author_column=author_column,
        text_column=text_column,
        datetime_column=datetime_column,
        source_name=source_name,
        metadata_column=metadata_column,
    )
