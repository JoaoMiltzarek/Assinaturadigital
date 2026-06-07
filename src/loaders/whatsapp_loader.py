from __future__ import annotations

from collections import Counter
import re

import pandas as pd

from src.processing.normalizer import normalize_messages_dataframe


_TIMESTAMP_CORE = (
    r"\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4},?\s+"
    r"\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s?[APap]\.?[Mm]\.?)?"
)

_RECORD_START_RE = re.compile(
    rf"(?:\[\s*)?{_TIMESTAMP_CORE}(?:\s*\])?\s*(?:-|–)?\s*"
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
    rf"\s*$",
    re.DOTALL,
)

INVISIBLE_CHARS = [
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\ufeff",
]

MEDIA_WORDS_EN = [
    "sticker",
    "image",
    "audio",
    "video",
    "gif",
    "document",
    "location",
    "contact card",
]

MEDIA_WORDS_PT = [
    "figurinha",
    "imagem",
    "áudio",
    "audio",
    "vídeo",
    "video",
    "documento",
    "localização",
    "localizacao",
    "mídia",
    "midia",
]

SYSTEM_MARKERS = [
    "messages and calls are end-to-end encrypted",
    "as mensagens e chamadas são protegidas",
    "as mensagens e as chamadas são protegidas",
    "changed their phone number",
    "mudou de número",
    "mudou de numero",
    "alterou o código de segurança",
    "alterou o codigo de seguranca",
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

EDITED_MARKERS = [
    "this message was edited",
    "<this message was edited>",
    "mensagem editada",
]

DELETED_MARKERS = [
    "this message was deleted",
    "<this message was deleted>",
    "mensagem apagada",
    "mensagem excluída",
    "mensagem excluida",
]


def _remove_invisible_chars(text: str) -> str:
    text = "" if text is None else str(text)

    for char in INVISIBLE_CHARS:
        text = text.replace(char, "")

    return text


def _normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_message_text(text: str) -> str:
    text = _remove_invisible_chars(text)
    text = _normalize_spaces(text)

    # Remove marcador de edição sem destruir a mensagem real.
    for marker in EDITED_MARKERS:
        text = re.sub(re.escape(marker), "", text, flags=re.IGNORECASE)

    # Remove links do texto usado para perfil.
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    return _normalize_spaces(text)


def _split_records(content: str) -> list[str]:
    """
    Divide o TXT em mensagens.

    Funciona melhor que split por linha porque WhatsApp pode exportar
    mensagens multilinha e também mensagens grudadas.
    """
    content = _remove_invisible_chars(content)
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    starts = [match.start() for match in _RECORD_START_RE.finditer(content)]

    if not starts:
        return [line.strip() for line in content.splitlines() if line.strip()]

    records = []

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(content)
        record = content[start:end].strip()

        if record:
            records.append(_normalize_spaces(record))

    return records


def _discard_reason(text: str) -> str | None:
    lower = _clean_message_text(text).lower().strip()

    if not lower:
        return "texto_vazio"

    if re.fullmatch(r"[\W_]+", lower):
        return "apenas_simbolos"

    for marker in DELETED_MARKERS:
        if marker in lower:
            return "mensagem_apagada"

    for word in MEDIA_WORDS_EN:
        if re.fullmatch(rf"<?{re.escape(word)} omitted>?", lower):
            return "midia_omitida"

    for word in MEDIA_WORDS_PT:
        if re.fullmatch(rf"<?{re.escape(word)} (omitid[ao]|oculta)>?", lower):
            return "midia_omitida"

    if "voice call" in lower or "video call" in lower:
        return "chamada"

    if "missed voice call" in lower or "missed video call" in lower:
        return "chamada"

    for marker in SYSTEM_MARKERS:
        if marker in lower:
            return "mensagem_de_sistema"

    return None


def load_whatsapp_txt_messages(
    text_content: str,
    source_name: str = "whatsapp",
    return_stats: bool = False,
):
    """
    Lê um TXT exportado do WhatsApp e retorna DataFrame normalizado.

    Se return_stats=True, retorna:
        (dataframe, stats)

    Caso contrário, retorna apenas:
        dataframe
    """
    records = _split_records(text_content)

    parsed_messages = []
    discard_reasons = Counter()

    for record in records:
        match = _MESSAGE_RE.match(record)

        if not match:
            discard_reasons["sem_autor_ou_formato_invalido"] += 1
            continue

        raw_datetime = match.group("datetime").replace(",", "").strip()
        author = match.group("author").strip()
        text = _clean_message_text(match.group("text"))

        reason = _discard_reason(text)

        if reason is not None:
            discard_reasons[reason] += 1
            continue

        parsed_messages.append(
            {
                "datetime": raw_datetime,
                "author": author,
                "text": text,
                "metadata": {
                    "raw_datetime": raw_datetime,
                    "loader": "whatsapp_txt",
                },
            }
        )

    raw_df = pd.DataFrame(
        parsed_messages,
        columns=["datetime", "author", "text", "metadata"],
    )

    normalized_df = normalize_messages_dataframe(
        raw_dataframe=raw_df,
        author_column="author",
        text_column="text",
        datetime_column="datetime",
        source_name=source_name,
        metadata_column="metadata",
    )

    stats = {
        "total_records_found": len(records),
        "valid_messages": len(normalized_df),
        "discarded_messages": sum(discard_reasons.values()),
        "discard_reasons": dict(discard_reasons),
    }

    if return_stats:
        return normalized_df, stats

    return normalized_df