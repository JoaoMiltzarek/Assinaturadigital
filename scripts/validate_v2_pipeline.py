import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH para permitir imports do pacote src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loaders.whatsapp_loader import load_whatsapp_txt_messages
from src.stylometry.profile_builder import build_author_stylometric_profile

def run_validation():
    fake_txt = (
        "31/05/2026 12:00 - Joao: mano kkk muito bom 😂\n"
        "31/05/2026 12:01 - Maria: verdade\n"
        "31/05/2026 12:02 - Joao: pior que é isso mesmo kkkkk\n"
        "31/05/2026 12:03 - Joao: vlw\n"
    )

    df = load_whatsapp_txt_messages(fake_txt)
    
    profile = build_author_stylometric_profile(df, "Joao")
    
    assert profile["message_count"] >= 2, f"message_count={profile['message_count']}"
    assert "top_words" in profile, "top_words missing"
    assert "top_bigrams" in profile, "top_bigrams missing"
    assert "sample_messages" in profile, "sample_messages missing"
    assert profile["total_laughter"] >= 1, f"total_laughter={profile['total_laughter']}"
    
    print("v2 pipeline ok")

if __name__ == "__main__":
    run_validation()
