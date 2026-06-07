from __future__ import annotations

import pandas as pd

def _validate_column(df: pd.DataFrame, column_name: str | None, context: str) -> None:
    if column_name is not None and column_name not in df.columns:
        raise ValueError(
            f"Coluna de {context} '{column_name}' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

def _parse_datetime_series_safely(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")

def normalize_messages_dataframe(
    raw_dataframe: pd.DataFrame,
    author_column: str,
    text_column: str,
    datetime_column: str | None = None,
    source_name: str = "unknown",
    metadata_column: str | None = None,
) -> pd.DataFrame:
    if not isinstance(raw_dataframe, pd.DataFrame):
        raise ValueError("raw_dataframe deve ser um pandas DataFrame.")
        
    _validate_column(raw_dataframe, author_column, "autor")
    _validate_column(raw_dataframe, text_column, "texto")
    _validate_column(raw_dataframe, datetime_column, "data/hora")
    _validate_column(raw_dataframe, metadata_column, "metadados")
    
    df = pd.DataFrame()
    
    df["author"] = raw_dataframe[author_column].astype(str).str.strip()
    df["text"] = raw_dataframe[text_column].astype(str).str.strip()
    
    if datetime_column is not None:
        df["datetime"] = _parse_datetime_series_safely(raw_dataframe[datetime_column])
    else:
        df["datetime"] = pd.NaT
        
    df["source"] = source_name
    
    if metadata_column is not None:
        df["metadata"] = raw_dataframe[metadata_column]
    else:
        df["metadata"] = [{} for _ in range(len(raw_dataframe))]
        
    valid_mask = (
        (df["author"] != "") & 
        (df["author"].str.lower() != "nan") & 
        (df["author"].str.lower() != "none") &
        (df["text"] != "") &
        (df["text"].str.lower() != "nan") &
        (df["text"].str.lower() != "none")
    )
    df = df[valid_mask].copy()
    
    expected_columns = ["author", "text", "datetime", "source", "metadata"]
    df = df[expected_columns]
    df.reset_index(drop=True, inplace=True)
    
    return df
