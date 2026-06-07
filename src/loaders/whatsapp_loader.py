"""
Loader para arquivos .txt exportados do WhatsApp.

Objetivo:
- Ler conversas exportadas do WhatsApp;
- Separar data/hora, autor e texto;
- Remover mídia omitida, chamadas, mensagens apagadas/editadas e mensagens de sistema;
- Retornar DataFrame normalizado para a V2.
"""

from __future__ import annotations

import re
import pandas as pd

from src.processing.normalizer import normalize_messages_dataframe


_TIMESTAMP_CORE = (
    r"\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s+"
    r"\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s?[APap]\.?[Mm]\.?)?"
)

_MESSAGE_START_RE = re.compile(
    rf"(?:\[\s*)?"
    rf"{_TIMESTAMP_CORE}"
    rf"(?:\s*\])?"
    rf"\s*(?:-|–)?\s*"
    rf"[^:\n]{{1,80}}"
    rf":\s*"
)

_MESSAGE_RE = re.compile(
    rf"^\s*"
    rf"(?:\[\s*)?"
    rf"(?P<datetime>{_TIMESTAMP_CORE})"
    rf"(?:\s*\])?"
    rf"\s*(?:-|–)?\s*"
    rf"(?P<author>[^:\n]{{1,80}})"
    rf":\s*"
    rf"(?P<text>.*)"
    rf"\s*$"
)


INVISIBLE_CHARS = [
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\ufeff",
]


NOISE_EXACT_MESSAGES = {
    "",
    "sticker omitted",
    "image omitted",
    "audio omitted",
    "video omitted",
    "gif omitted",
    "document omitted",
    "contact card omitted",
    "location omitted",
    "voice call",
    "video call",
    "missed voice call",
    "missed video call",
    "this message was deleted",
    "this message was edited",
    "<this message was edited>",
    "<this message was deleted>",
    "imagem omitida",
    "áudio omitido",
    "audio omitido",
    "vídeo omitido",
    "video omitido",
    "figurinha omitida",
    "documento omitido",
    "cartão de contato omitido",
    "cartao de contato omitido",
    "localização omitida",
    "localizacao omitida",
    "mídia oculta",
    "midia oculta",
    "<mídia oculta>",
    "<midia oculta>",
}


NOISE_CONTAINS = [
    "messages and calls are end-to-end encrypted",
    "as mensagens e chamadas são protegidas",
    "as mensagens e as chamadas são protegidas",
    "changed their phone number",
    "mudou de número",
    "alterou o código de segurança",
    "changed the subject",
    "changed this group's icon",
    "created group",
    "criou o grupo",
    "adicionou",
    "removeu",
    "saiu",
    "left",
    "joined using this group's invite link",
]


def _remove_invisible_chars(content: str) -> str:
    content = str(content)

    for char in INVISIBLE_CHARS:
        content = content.replace(char, "")

    return content


def _normalize_whatsapp_text(text: str) -> str:
    """
    Limpa o texto da mensagem sem destruir estilo.
    Mantém pontuação, emojis e gírias reais.
    Remove lixo técnico do WhatsApp.
    """
    text = str(text).strip()

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove marcadores de mensagem editada/apagada.
    text = re.sub(r"<?this message was edited>?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<?this message was deleted>?", "", text, flags=re.IGNORECASE)

    # Remove links, porque eles poluem o vocabulário.
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def _is_noise_message(text: str) -> bool:
    """
    Decide se uma mensagem deve ser descartada.
    """
    text = _normalize_whatsapp_text(text)
    lower = text.lower().strip()

    if lower in NOISE_EXACT_MESSAGES:
        return True

    if not lower:
        return True

    # Remove mídia omitida em inglês.
    if re.fullmatch(r"(sticker|image|audio|video|gif|document|location|contact card)\s+omitted", lower):
        return True

    # Remove mídia omitida em português.
    if re.fullmatch(r"(imagem|áudio|audio|vídeo|video|figurinha|documento|localização|localizacao)\s+omitid[ao]", lower):
        return True

    # Remove chamadas.
    if "voice call" in lower or "video call" in lower:
        return True

    # Remove mensagens de sistema.
    if any(marker in lower for marker in NOISE_CONTAINS):
        return True

    # Remove mensagem que é só pontuação/símbolo.
    if re.fullmatch(r"[\W_]+", lower):
        return True

    return False


def _split_whatsapp_records(content: str) -> list[str]:
    """
    Divide o arquivo em mensagens.

    Em vez de ler linha por linha, encontra todos os inícios de mensagem.
    Isso evita quebrar mensagens multilinha e também resolve exportações grudadas.
    """
    content = _remove_invisible_chars(content)
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    starts = [match.start() for match in _MESSAGE_START_RE.finditer(content)]

    if not starts:
        return [line.strip() for line in content.splitlines() if line.strip()]

    records = []

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(content)
        record = content[start:end].strip()

        if record:
            record = re.sub(r"\s*\n\s*", " ", record)
            record = re.sub(r"\s+", " ", record).strip()
            records.append(record)

    return records


def load_whatsapp_txt_messages(
    text_content: str,
    source_name: str = "whatsapp",
) -> pd.DataFrame:
    """
    Lê um TXT exportado do WhatsApp e retorna DataFrame normalizado.

    Colunas finais:
    - author
    - text
    - datetime
    - source
    - metadata
    """
    records = _split_whatsapp_records(text_content)

    parsed_messages = []

    for record in records:
        match = _MESSAGE_RE.match(record)

        if not match:
            continue

        author = match.group("author").strip()
        text = _normalize_whatsapp_text(match.group("text"))

        if _is_noise_message(text):
            continue

        parsed_messages.append(
            {
                "datetime": match.group("datetime").replace(",", "").strip(),
                "author": author,
                "text": text,
            }
        )

    raw_df = pd.DataFrame(parsed_messages, columns=["datetime", "author", "text"])

    return normalize_messages_dataframe(
        raw_dataframe=raw_df,
        author_column="author",
        text_column="text",
        datetime_column="datetime",
        source_name=source_name,
    )