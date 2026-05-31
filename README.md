# AssinaturaDigital

Projeto desenvolvido para a disciplina de Introdução à IA e Ciência de Dados — Unisinos.

## Objetivo

O objetivo do projeto é identificar automaticamente o autor de um tweet com base no estilo de escrita.

O sistema utiliza técnicas de estilometria e Machine Learning para extrair características do texto, como:

- comprimento médio das palavras;
- número de palavras;
- quantidade de pontuação;
- uso de maiúsculas;
- emojis;
- hashtags;
- menções;
- exclamações;
- interrogações.

## Dataset

O dataset utilizado é o Tweets Data for Authorship Attribution Modelling, disponível no Kaggle.

O arquivo principal é:

```text
data/tweet.csv
```

## Versão 2.0 — Perfilador Estilométrico Universal

A v2 permite criar perfil estilométrico a partir de CSV, texto simples ou exportação TXT do WhatsApp.

**Formato padrão interno:**
`author`, `text`, `datetime`, `source`, `metadata`

**Como rodar:**
```bash
python -m streamlit run app.py
```

**Como testar os dados de exemplo:**
```bash
python scripts\validate_v2_sample_data.py
```

*Nota: a recriação de estilo ainda não está implementada.*

> **Nota Ética:** o sistema deve ser usado com dados próprios, públicos ou autorizados, e não para falsificação de identidade.