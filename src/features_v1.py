import re
import string
import emoji


FEATURES_V1 = [
    "avg_word_length",
    "num_words",
    "num_unique_words",
    "unique_ratio",
    "num_chars",
    "num_punctuation",
    "num_uppercase",
    "num_emojis",
    "num_hashtags",
    "num_mentions",
    "num_exclamation",
    "num_question",
]


def limpar_texto(texto):
    if texto is None:
        return ""

    texto = str(texto)
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"#", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def contar_emojis(texto):
    if texto is None:
        return 0

    return sum(1 for char in str(texto) if char in emoji.EMOJI_DATA)


def extrair_features_v1(texto_original):
    texto_original = str(texto_original)
    texto_limpo = limpar_texto(texto_original)

    palavras = texto_limpo.split()
    total_palavras = len(palavras)

    tamanhos_palavras = [len(p) for p in palavras]

    avg_word_length = sum(tamanhos_palavras) / total_palavras if total_palavras > 0 else 0
    num_words = total_palavras
    num_unique_words = len(set(palavras))
    unique_ratio = num_unique_words / total_palavras if total_palavras > 0 else 0

    features = {
        "avg_word_length": avg_word_length,
        "num_words": num_words,
        "num_unique_words": num_unique_words,
        "unique_ratio": unique_ratio,
        "num_chars": len(texto_limpo),
        "num_punctuation": sum(1 for c in texto_original if c in string.punctuation),
        "num_uppercase": sum(1 for c in texto_original if c.isupper()),
        "num_emojis": contar_emojis(texto_original),
        "num_hashtags": len(re.findall(r"#\w+", texto_original)),
        "num_mentions": len(re.findall(r"@\w+", texto_original)),
        "num_exclamation": texto_original.count("!"),
        "num_question": texto_original.count("?"),
    }

    return features