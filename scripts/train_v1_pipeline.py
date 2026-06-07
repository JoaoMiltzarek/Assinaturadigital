from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.v1_text_pipeline import build_v1_pipeline


DATA_PATH = PROJECT_ROOT / "data" / "tweet.csv"
MODELS_DIR = PROJECT_ROOT / "models"

PIPELINE_PATH = MODELS_DIR / "v1_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "v1_metrics.csv"
CONFUSION_MATRIX_PATH = MODELS_DIR / "v1_confusion_matrix.csv"
SUMMARY_PATH = MODELS_DIR / "v1_model_summary.txt"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    expected_columns = {"Name", "Tweet description"}
    missing = expected_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Colunas esperadas não encontradas: {missing}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df = df.rename(
        columns={
            "Name": "author",
            "Tweet description": "text",
        }
    )

    df = df[["author", "text"]].copy()
    df = df.dropna(subset=["author", "text"])

    df["author"] = df["author"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    df = df[(df["author"] != "") & (df["text"] != "")].copy()

    return df


def evaluate_model(model_name: str, pipeline, x_test, y_test) -> dict[str, float | str]:
    predictions = pipeline.predict(x_test)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_dataset()

    print(f"Dataset carregado: {len(df)} linhas")
    print(f"Autores encontrados: {df['author'].nunique()}")

    x = df["text"]
    y = df["author"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    candidate_models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
        "complement_nb": ComplementNB(),
    }

    trained_pipelines = {}
    metrics = []

    for model_name, classifier in candidate_models.items():
        print(f"\nTreinando: {model_name}")

        pipeline = build_v1_pipeline(classifier)
        pipeline.fit(x_train, y_train)

        result = evaluate_model(model_name, pipeline, x_test, y_test)
        metrics.append(result)
        trained_pipelines[model_name] = pipeline

        print(result)

    metrics_df = pd.DataFrame(metrics).sort_values(
        by="f1_macro",
        ascending=False,
    )

    best_model_name = metrics_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\nMelhor modelo:")
    print(metrics_df.iloc[0])

    joblib.dump(best_pipeline, PIPELINE_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)

    labels = sorted(y.unique().tolist())
    best_predictions = best_pipeline.predict(x_test)

    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write(f"Melhor modelo: {best_model_name}\n")
        file.write(f"Treino: {len(x_train)} exemplos\n")
        file.write(f"Teste: {len(x_test)} exemplos\n")
        file.write("\nMétricas:\n")
        file.write(metrics_df.to_string(index=False))

    print(f"\nPipeline salvo em: {PIPELINE_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(f"Matriz de confusão salva em: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
