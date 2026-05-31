"""
src/processing/normalizer.py

Responsável por normalizar qualquer DataFrame de mensagens (vindo de CSV,
JSON, WhatsApp, etc) para o formato interno padrão do AssinaturaDigital.

Formato padrão (obrigatório após a normalização):
- author (str)
- text (str)
- datetime (datetime64 ou NaT)
- source (str)
- metadata (object/dict)
"""

from __future__ import annotations

import pandas as pd
import numpy as np

def _validate_column(df: pd.DataFrame, column_name: str | None, context: str) -> None:
    """Valida se a coluna existe no DataFrame, levantando ValueError se não existir."""
    if column_name is not None and column_name not in df.columns:
        raise ValueError(
            f"Coluna de {context} '{column_name}' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

def _parse_datetime_series_safely(series: pd.Series) -> pd.Series:
    """Faz o parsing de datas suprimindo warnings de formato misto/ambíguo."""
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")

def normalize_messages_dataframe(
    raw_dataframe: pd.DataFrame,
    author_column: str,
    text_column: str,
    datetime_column: str | None = None,
    source_name: str = "unknown",
    metadata_column: str | None = None,
) -> pd.DataFrame:
    """
    Transforma um DataFrame bruto no formato padrão do AssinaturaDigital.
    
    Remove linhas em que o autor ou o texto fiquem vazios após limpeza.
    Não altera o DataFrame original.
    
    Args:
        raw_dataframe: O DataFrame de origem.
        author_column: Nome da coluna que contém o autor.
        text_column: Nome da coluna que contém a mensagem.
        datetime_column: Nome da coluna com a data/hora (opcional).
        source_name: String constante indicando a origem (ex: 'whatsapp').
        metadata_column: Nome da coluna com metadados adicionais (opcional).
        
    Returns:
        Um novo DataFrame com exatamente as colunas:
        ['author', 'text', 'datetime', 'source', 'metadata'].
        
    Raises:
        ValueError: Se raw_dataframe não for um pd.DataFrame ou se as colunas
            especificadas não existirem.
    """
    if not isinstance(raw_dataframe, pd.DataFrame):
        raise ValueError("raw_dataframe deve ser um pandas DataFrame.")
        
    # Validações de colunas
    _validate_column(raw_dataframe, author_column, "autor")
    _validate_column(raw_dataframe, text_column, "texto")
    _validate_column(raw_dataframe, datetime_column, "data/hora")
    _validate_column(raw_dataframe, metadata_column, "metadados")
    
    # Criar novo DataFrame para não alterar o original
    df = pd.DataFrame()
    
    # Converter para string de forma segura
    df["author"] = raw_dataframe[author_column].astype(str).str.strip()
    df["text"] = raw_dataframe[text_column].astype(str).str.strip()
    
    # Preencher datetime
    if datetime_column is not None:
        df["datetime"] = _parse_datetime_series_safely(raw_dataframe[datetime_column])
    else:
        df["datetime"] = pd.NaT
        
    # Preencher source
    df["source"] = source_name
    
    # Preencher metadata
    if metadata_column is not None:
        df["metadata"] = raw_dataframe[metadata_column]
    else:
        # Usar list comprehension ou assign para garantir um dict vazio independente por linha
        df["metadata"] = [{} for _ in range(len(raw_dataframe))]
        
    # Filtrar linhas inválidas (autor vazio ou texto vazio)
    # Consideramos 'nan' (string resultante de np.nan) como vazio também
    valid_mask = (
        (df["author"] != "") & 
        (df["author"].str.lower() != "nan") & 
        (df["author"].str.lower() != "none") &
        (df["text"] != "") &
        (df["text"].str.lower() != "nan") &
        (df["text"].str.lower() != "none")
    )
    df = df[valid_mask].copy()
    
    # Reordenar colunas pro padrão estrito e resetar índice
    expected_columns = ["author", "text", "datetime", "source", "metadata"]
    df = df[expected_columns]
    df.reset_index(drop=True, inplace=True)
    
    return df
