"""
src/loaders/whatsapp_loader.py

Loader para arquivos .txt exportados do WhatsApp.
Usa expressões regulares para detectar a data/hora, autor e mensagem.
Lida com mensagens com quebra de linha.
"""

from __future__ import annotations

import re
import pandas as pd

from src.processing.normalizer import normalize_messages_dataframe

# Padrão Android PT-BR: "31/05/2026 12:00 - Joao: oi kkk"
_ANDROID_PATTERN = re.compile(r"^(\d{2}/\d{2}/\d{4}\s\d{2}:\d{2})\s-\s(.*?):\s(.*)$")

# Padrão iOS/Colchetes: "[31/05/2026, 12:00:00] Joao: oi kkk"
_IOS_PATTERN = re.compile(r"^\[(\d{2}/\d{2}/\d{4},\s\d{2}:\d{2}:\d{2})\]\s(.*?):\s(.*)$")

def load_whatsapp_txt_messages(
    text_content: str,
    source_name: str = "whatsapp",
) -> pd.DataFrame:
    """
    Transforma texto exportado do WhatsApp em um DataFrame normalizado.
    
    Tenta capturar o formato Android ou iOS/Colchetes. Se uma linha não for
    reconhecida como nova mensagem (sem autor/data), ela é anexada à 
    mensagem imediatamente anterior.
    
    Args:
        text_content: Conteúdo completo do arquivo .txt.
        source_name: Nome da fonte.
        
    Returns:
        DataFrame padronizado.
    """
    lines = text_content.splitlines()
    
    parsed_messages = []
    current_msg = None
    
    for line in lines:
        if not line.strip():
            continue
            
        # Tenta casar com Android
        match_android = _ANDROID_PATTERN.match(line)
        if match_android:
            if current_msg:
                parsed_messages.append(current_msg)
            current_msg = {
                "datetime": match_android.group(1),
                "author": match_android.group(2).strip(),
                "text": match_android.group(3).strip(),
            }
            continue
            
        # Tenta casar com iOS
        match_ios = _IOS_PATTERN.match(line)
        if match_ios:
            if current_msg:
                parsed_messages.append(current_msg)
            current_msg = {
                "datetime": match_ios.group(1).replace(",", ""),
                "author": match_ios.group(2).strip(),
                "text": match_ios.group(3).strip(),
            }
            continue
            
        # Se não casou com nenhum, é continuação da mensagem anterior (ou sistema)
        if current_msg is not None:
            current_msg["text"] += "\n" + line.strip()
            
    # Adiciona a última mensagem pendente
    if current_msg:
        parsed_messages.append(current_msg)
        
    # Cria DataFrame bruto
    if parsed_messages:
        raw_df = pd.DataFrame(parsed_messages)
    else:
        raw_df = pd.DataFrame(columns=["datetime", "author", "text"])
        
    return normalize_messages_dataframe(
        raw_dataframe=raw_df,
        author_column="author",
        text_column="text",
        datetime_column="datetime",
        source_name=source_name,
    )
