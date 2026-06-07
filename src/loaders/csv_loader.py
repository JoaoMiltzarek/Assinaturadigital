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
    raw_df = pd.read_csv(csv_path_or_buffer)
    
    return normalize_messages_dataframe(
        raw_dataframe=raw_df,
        author_column=author_column,
        text_column=text_column,
        datetime_column=datetime_column,
        source_name=source_name,
        metadata_column=metadata_column,
    )
