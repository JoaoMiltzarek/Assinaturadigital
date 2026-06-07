# AssinaturaDigital

Projeto acadêmico de Machine Learning aplicado à **estilometria digital**.

O objetivo é extrair atributos de escrita a partir de textos, treinar modelos de classificação de autoria e gerar perfis estilométricos de autores.

---

## Estrutura do projeto

O projeto é dividido em duas partes:

**V1 — Classificador de autoria:** recebe um texto e prevê o autor provável com base em features extraídas e um modelo treinado.

**V2 — Perfilador estilométrico:** carrega mensagens de um autor (texto, CSV ou exportação de WhatsApp), gera um perfil de escrita e aplica uma recriação simples baseada em regras.

---

## Dataset

```
data/tweet.csv
```

Colunas utilizadas: `Name` (classe) e `Tweet description` (entrada). Registros com valores ausentes nessas colunas são removidos antes do treinamento.

---

## Features — V1

**Features manuais** (`src/features_v1.py`): tamanho médio de palavras, número de palavras, proporção de palavras únicas, número de caracteres, pontuação, proporção de maiúsculas, emojis, hashtags, menções, exclamações e interrogações.

**TF-IDF de caracteres** (`src/models/v1_text_pipeline.py`): n-grams de caracteres para capturar padrões de digitação, abreviações, risadas e pontuação repetida.

As features manuais são normalizadas com `StandardScaler`. O pipeline completo é salvo em `models/v1_pipeline.pkl`.

---

## Modelos e resultados

Três modelos foram comparados usando as métricas accuracy, precision macro, recall macro e F1 macro:

| Modelo | Accuracy | F1 macro |
|---|---|---|
| Logistic Regression | — | — |
| **Linear SVM** | **0.8469** | **0.8451** |
| Complement Naive Bayes | — | — |

O Linear SVM obteve o melhor F1 macro e foi selecionado como modelo final. As métricas completas estão em `models/v1_metrics.csv`.

---

## Interface

```bash
python -m streamlit run app.py
```

A interface possui duas abas: **Classificador** (inferência de autoria) e **Perfilador** (análise estilométrica e recriação de estilo).

---

## Instalação

```bash
# Ativar ambiente virtual
.\venv\Scripts\Activate.ps1

# Instalar dependências
python -m pip install -r requirements.txt
```

---

## Treinamento e validação

```bash
# Treinar
python scripts\train_v1_pipeline.py

# Validar
python -m py_compile app.py
python scripts\validate_v1_pipeline.py
python scripts\validate_v2_pipeline.py
python scripts\validate_v2_sample_data.py
python scripts\validate_whatsapp_edge_cases.py
```

---

## Limitações

Textos muito curtos reduzem a confiabilidade da classificação. Autores com estilos semelhantes tendem a ser confundidos. A V2 é experimental — a recriação de estilo é aproximada e restrita a padrões realmente presentes no perfil do autor. O sistema não deve ser usado para identificação de pessoas fora do contexto acadêmico.

---

## Estrutura de pastas

```
AssinaturaDigital/
├── app.py
├── requirements.txt
├── data/tweet.csv
├── models/
│   ├── v1_pipeline.pkl
│   ├── v1_metrics.csv
│   ├── v1_confusion_matrix.csv
│   └── v1_model_summary.txt
├── scripts/
│   ├── train_v1_pipeline.py
│   ├── validate_v1_pipeline.py
│   ├── validate_v2_pipeline.py
│   ├── validate_v2_sample_data.py
│   └── validate_whatsapp_edge_cases.py
└── src/
    ├── features_v1.py
    ├── models/v1_text_pipeline.py
    ├── loaders/
    │   ├── csv_loader.py
    │   ├── generic_text_loader.py
    │   └── whatsapp_loader.py
    ├── processing/normalizer.py
    └── stylometry/
        ├── profile_builder.py
        └── style_replicator.py
```