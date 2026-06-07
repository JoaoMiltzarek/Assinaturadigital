from __future__ import annotations

import pandas as pd

from src.processing.normalizer import normalize_messages_dataframe

def load_generic_text_messages(
    text_content: str,
    author_name: str = "unknown_author",
    source_name: str = "text",
) -> pd.DataFrame:
    lines = [line.strip() for line in text_content.splitlines() if line.strip()]
    
    raw_df = pd.DataFrame({
        "author": [author_name] * len(lines),
        "text": lines,
    })
    
    if raw_df.empty:
        raw_df = pd.DataFrame(columns=["author", "text"])
        
    return normalize_messages_dataframe(
        raw_dataframe=raw_df,
        author_column="author",
        text_column="text",
        source_name=source_name,
    )
