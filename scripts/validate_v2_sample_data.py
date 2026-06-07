import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loaders.csv_loader import load_csv_messages
from src.loaders.generic_text_loader import load_generic_text_messages
from src.loaders.whatsapp_loader import load_whatsapp_txt_messages
from src.stylometry.profile_builder import build_author_stylometric_profile

def validate_dataframe(df, expected_min_len, source_type):
    assert list(df.columns) == ['author', 'text', 'datetime', 'source', 'metadata'], f"Colunas erradas em {source_type}"
    assert len(df) >= expected_min_len, f"Tamanho insuficiente em {source_type}"

def validate_profile(profile, expected_min_msgs, source_type):
    assert profile['message_count'] >= expected_min_msgs, f"Mensagens insuficientes em {source_type}"
    assert profile['total_laughter'] >= 2, f"Total laughter insuficiente em {source_type}"
    assert 'top_words' in profile, f"top_words faltando em {source_type}"
    assert 'top_bigrams' in profile, f"top_bigrams faltando em {source_type}"
    assert 'sample_messages' in profile, f"sample_messages faltando em {source_type}"

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # CSV
    csv_path = os.path.join(base_dir, 'sample_data', 'messages_example.csv')
    df_csv = load_csv_messages(csv_path, author_column='author', text_column='text', datetime_column='datetime')
    validate_dataframe(df_csv, 5, "CSV")
    prof_csv = build_author_stylometric_profile(df_csv, "Joao")
    validate_profile(prof_csv, 3, "CSV")
    
    # Texto simples
    txt_path = os.path.join(base_dir, 'sample_data', 'generic_text_example.txt')
    with open(txt_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    df_txt = load_generic_text_messages(text_content, author_name="Joao")
    validate_dataframe(df_txt, 5, "Generic Text")
    prof_txt = build_author_stylometric_profile(df_txt, "Joao")
    validate_profile(prof_txt, 5, "Generic Text")
    
    # WhatsApp
    wpp_path = os.path.join(base_dir, 'sample_data', 'whatsapp_example.txt')
    with open(wpp_path, 'r', encoding='utf-8') as f:
        wpp_content = f.read()
    df_wpp = load_whatsapp_txt_messages(wpp_content)
    validate_dataframe(df_wpp, 5, "WhatsApp")
    prof_wpp = build_author_stylometric_profile(df_wpp, "Joao")
    validate_profile(prof_wpp, 3, "WhatsApp")
    
    print("v2 sample data ok")

if __name__ == "__main__":
    main()
