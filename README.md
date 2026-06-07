# AssinaturaDigital

Projeto acadêmico de Machine Learning sobre **estilometria aplicada à escrita digital**.

O objetivo do projeto é analisar padrões de escrita em textos digitais, identificar características estilométricas de autores e aplicar essas informações em duas etapas principais:

1. **Classificação de autoria**: prever qual autor provavelmente escreveu um texto.
2. **Perfil estilométrico**: gerar um perfil de escrita a partir de mensagens e aplicar uma recriação simples baseada em regras.

---

## 1. Objetivo do projeto

O projeto busca estudar como padrões de escrita podem ser representados por atributos numéricos e usados em modelos de Machine Learning.

Entre os padrões analisados estão:

* tamanho médio das palavras;
* quantidade de palavras;
* quantidade de palavras únicas;
* proporção de palavras únicas;
* número de caracteres;
* uso de pontuação;
* uso de letras maiúsculas;
* emojis;
* hashtags;
* menções;
* exclamações;
* interrogações;
* padrões de caracteres por TF-IDF.

---

## 2. Estrutura geral

O projeto possui duas partes principais:

## V1 — Classificador de autoria

A V1 é a parte principal de Machine Learning do projeto.

Ela recebe um texto e tenta prever qual autor provavelmente escreveu aquele conteúdo.

A abordagem final combina:

* features manuais de estilometria;
* TF-IDF com n-grams de caracteres;
* comparação entre três modelos de classificação;
* escolha do melhor modelo com base em métricas de avaliação;
* inferência com textos novos pela interface Streamlit.

O pipeline final fica salvo em:

```text
models/v1_pipeline.pkl
```

Esse pipeline contém todas as etapas necessárias para a inferência, evitando diferença entre o pré-processamento usado no treino e o usado no app.

---

## V2 — Perfilador estilométrico

A V2 é uma extensão experimental do projeto.

Ela permite carregar mensagens e gerar um perfil de escrita de um autor selecionado.

Entradas aceitas:

* texto simples;
* arquivo CSV;
* conversa exportada do WhatsApp em `.txt`.

A V2 calcula informações como:

* palavras frequentes;
* bigramas frequentes;
* risadas;
* emojis;
* pontuação;
* tamanho médio das mensagens;
* exemplos reais de mensagens do autor.

Também existe uma recriação simples de mensagem baseada no perfil gerado.

Essa recriação é feita por regras e tem caráter aproximado. Ela preserva o sentido da mensagem original e só aplica padrões encontrados no perfil do autor.

---

## 3. Dataset

A V1 utiliza o arquivo:

```text
data/tweet.csv
```

O dataset contém textos associados a autores. Ele é usado para treinar o classificador de autoria.

As colunas principais usadas no treinamento são:

* `Name`: classe/autoria;
* `Tweet description`: texto usado como entrada.

Durante o treinamento, o projeto remove registros inválidos ou vazios nas colunas necessárias.

---

## 4. Features usadas na V1

A V1 utiliza dois grupos principais de atributos.

### 4.1. Features manuais

Extraídas pela função:

```text
src/features_v1.py
```

Principais atributos:

* tamanho médio das palavras;
* número de palavras;
* número de palavras únicas;
* proporção de palavras únicas;
* número de caracteres;
* quantidade de pontuação;
* proporção de letras maiúsculas;
* número de emojis;
* número de hashtags;
* número de menções;
* número de exclamações;
* número de interrogações.

### 4.2. TF-IDF com n-grams de caracteres

Implementado em:

```text
src/models/v1_text_pipeline.py
```

O TF-IDF de caracteres permite capturar padrões pequenos de escrita, como:

* abreviações;
* risadas;
* pontuação repetida;
* combinações de caracteres;
* estilo informal;
* padrões recorrentes de digitação.

---

## 5. Modelos treinados

O treinamento da V1 compara três modelos:

* Logistic Regression;
* Linear SVM;
* Complement Naive Bayes.

O script de treino está em:

```text
scripts/train_v1_pipeline.py
```

As métricas usadas são:

* accuracy;
* precision macro;
* recall macro;
* F1 macro.

A métrica macro é importante porque avalia o desempenho considerando todas as classes, reduzindo o risco de olhar apenas para o acerto geral.

---

## 6. Resultados da V1

Após os testes, o melhor modelo foi:

```text
Linear SVM
```

Resultados salvos:

```text
models/v1_metrics.csv
models/v1_confusion_matrix.csv
models/v1_model_summary.txt
```

O resultado final registrado foi aproximadamente:

```text
Accuracy: 0.8469
F1 macro: 0.8451
```

Esses valores indicam que o modelo conseguiu capturar padrões relevantes de autoria a partir dos textos do dataset.

---

## 7. Interface Streamlit

A interface principal está no arquivo:

```text
app.py
```

Para rodar o app:

```powershell
python -m streamlit run app.py
```

A interface possui duas abas:

1. **Classificador**
   Permite inserir um texto e prever a autoria provável.

2. **Perfilador**
   Permite carregar mensagens, selecionar um autor e gerar um perfil estilométrico.

---

## 8. Como instalar e rodar

### 8.1. Criar ou ativar ambiente virtual

No PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Ou, caso esteja usando `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 8.2. Instalar dependências

```powershell
python -m pip install -r requirements.txt
```

### 8.3. Rodar o app

```powershell
python -m streamlit run app.py
```

---

## 9. Como treinar a V1

Para treinar novamente o pipeline da V1:

```powershell
python scripts\train_v1_pipeline.py
```

Esse comando gera ou atualiza os arquivos:

```text
models/v1_pipeline.pkl
models/v1_metrics.csv
models/v1_confusion_matrix.csv
models/v1_model_summary.txt
```

---

## 10. Como validar o projeto

Rodar os testes principais:

```powershell
python -m py_compile app.py
python -m py_compile src\stylometry\style_replicator.py
python scripts\validate_v1_pipeline.py
python scripts\validate_v2_pipeline.py
python scripts\validate_v2_sample_data.py
python scripts\validate_whatsapp_edge_cases.py
```

Resultado esperado:

* o app deve compilar sem erro;
* o pipeline da V1 deve carregar e prever;
* os loaders da V2 devem funcionar;
* o perfil estilométrico deve ser gerado;
* os casos de WhatsApp devem ser tratados.

---

## 11. Limpeza e tratamento da V2

O loader de WhatsApp tenta remover ruídos comuns de conversas exportadas, como:

* mídia omitida;
* mensagens apagadas;
* links;
* chamadas;
* mensagens de sistema;
* caracteres invisíveis;
* linhas vazias;
* textos sem autor válido.

O perfilador também evita contar como palavras relevantes:

* datas;
* horários;
* nomes dos autores;
* tokens vazios;
* pontuação solta;
* termos técnicos de mensagens exportadas.

---

## 12. Recriação de estilo

A recriação de estilo da V2 é simples e baseada em regras.

Ela pode adaptar:

* maiúsculas/minúsculas;
* pontuação;
* abreviações;
* risadas;
* emojis.

Porém, o sistema só aplica padrões encontrados no perfil real do autor.

Exemplos:

* só usa `hj` se `hj` apareceu nas mensagens do autor;
* só usa `vc` se `vc` apareceu nas mensagens do autor;
* só adiciona risada se risadas apareceram no perfil;
* só adiciona emoji se emojis apareceram nos exemplos reais.

Essa etapa não busca gerar uma imitação perfeita. Ela serve para demonstrar como um perfil estilométrico pode orientar uma adaptação simples de texto.

---

## 13. Relação com os requisitos do trabalho

| Requisito                   | Onde aparece no projeto                           |
| --------------------------- | ------------------------------------------------- |
| Achar e baixar dados        | Dataset `data/tweet.csv`                          |
| Importar dados              | Leitura com Pandas                                |
| Analisar dados              | Contagem de autores, textos e métricas            |
| Entender e explicar dados   | README, app e análise das features                |
| Usar gráficos               | Métricas, matriz de confusão e gráficos no app    |
| Corrigir dados inválidos    | Remoção de textos/autores vazios                  |
| Corrigir dados faltantes    | `dropna` nas colunas essenciais                   |
| Decidir atributos úteis     | Features manuais + TF-IDF char n-grams            |
| Classificar variáveis       | Texto como entrada e autor como classe            |
| Separar previsores e classe | `x = text`, `y = author`                          |
| Transformar colunas         | TF-IDF e extração de features                     |
| Padronizar dados            | StandardScaler nas features manuais               |
| Dividir treino/teste        | `train_test_split`                                |
| Treinar 3 modelos           | Logistic Regression, Linear SVM, ComplementNB     |
| Calcular métricas           | Accuracy, precision macro, recall macro, F1 macro |
| Escolher melhor modelo      | Melhor F1 macro                                   |
| Inferência nova             | Interface Streamlit                               |
| Interface                   | `app.py`                                          |
| Git                         | Projeto versionado no repositório                 |

---

## 14. Limitações

O projeto possui algumas limitações importantes:

* textos muito curtos podem ser difíceis de classificar;
* o desempenho depende da qualidade e quantidade dos dados;
* autores com estilos parecidos podem ser confundidos;
* a V2 é uma extensão experimental;
* a recriação de estilo não é uma geração perfeita;
* o sistema não deve ser usado para identificação real de pessoas sem contexto ou validação adequada.

---

## 15. Estrutura de pastas

```text
Assinaturadigital-1/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── tweet.csv
│
├── models/
│   ├── v1_pipeline.pkl
│   ├── v1_metrics.csv
│   ├── v1_confusion_matrix.csv
│   └── v1_model_summary.txt
│
├── scripts/
│   ├── train_v1_pipeline.py
│   ├── validate_v1_pipeline.py
│   ├── validate_v2_pipeline.py
│   ├── validate_v2_sample_data.py
│   └── validate_whatsapp_edge_cases.py
│
└── src/
    ├── features_v1.py
    │
    ├── models/
    │   └── v1_text_pipeline.py
    │
    ├── loaders/
    │   ├── csv_loader.py
    │   ├── generic_text_loader.py
    │   └── whatsapp_loader.py
    │
    ├── processing/
    │   └── normalizer.py
    │
    └── stylometry/
        ├── profile_builder.py
        └── style_replicator.py
```

---

## 16. Arquivos antigos

Os arquivos abaixo pertencem ao fluxo anterior da V1 e podem existir apenas como backup:

```text
models/modelo_final.pkl
models/scaler.pkl
models/features.pkl
```

O fluxo principal atual usa:

```text
models/v1_pipeline.pkl
```

---

## 17. Conclusão

O projeto demonstra uma aplicação de Machine Learning em estilometria digital.

A V1 atende ao núcleo do trabalho ao treinar, comparar e avaliar modelos de classificação de autoria. A V2 complementa o projeto com uma análise interpretável de mensagens e uma adaptação simples baseada no perfil estilométrico do autor.

O foco principal da entrega é mostrar como padrões de escrita podem ser transformados em atributos, avaliados por modelos de Machine Learning e usados em uma interface prática.
