import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loaders.whatsapp_loader import load_whatsapp_txt_messages
from src.stylometry.profile_builder import build_author_stylometric_profile


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    whatsapp_path = os.path.join(base_dir, "sample_data", "whatsapp_edge_cases.txt")

    with open(whatsapp_path, "r", encoding="utf-8") as file:
        whatsapp_content = file.read()

    dataframe = load_whatsapp_txt_messages(whatsapp_content)

    assert len(dataframe) >= 5, f"Quantidade insuficiente de mensagens: {len(dataframe)}"
    assert "Joao" in dataframe["author"].values, "Autor Joao nao encontrado"
    assert "Maria" in dataframe["author"].values, "Autor Maria nao encontrado"
    assert "essa linha continua" in dataframe.iloc[2]["text"], "Continuacao nao anexada"

    profile = build_author_stylometric_profile(dataframe, "Joao")

    assert profile["message_count"] >= 3, f"message_count={profile['message_count']}"
    assert profile["total_laughter"] >= 2, f"total_laughter={profile['total_laughter']}"
    assert "top_words" in profile, "top_words missing"
    assert "top_bigrams" in profile, "top_bigrams missing"
    assert "sample_messages" in profile, "sample_messages missing"

    print("whatsapp edge cases ok")


if __name__ == "__main__":
    main()
