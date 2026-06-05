"""
Treinamento leve de modelos para identificacao universal de autoria.

O modulo treina modelos em memoria e nao salva artefatos em disco.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from src.stylometry.feature_extractor import extract_stylometric_features

_REQUIRED_COLUMNS = {"author", "text"}


def _filter_valid_messages(
    messages_dataframe: pd.DataFrame,
    min_messages_per_author: int,
) -> pd.DataFrame:
    """Remove mensagens invalidas e autores sem volume minimo."""
    if not isinstance(messages_dataframe, pd.DataFrame):
        raise ValueError("messages_dataframe deve ser um pandas DataFrame.")

    missing_columns = _REQUIRED_COLUMNS - set(messages_dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"O DataFrame nao contem as colunas obrigatorias: {missing_columns}."
        )

    if min_messages_per_author < 1:
        raise ValueError("min_messages_per_author deve ser maior ou igual a 1.")

    filtered_dataframe = messages_dataframe[["author", "text"]].copy()
    filtered_dataframe.dropna(subset=["author", "text"], inplace=True)
    filtered_dataframe["author"] = filtered_dataframe["author"].astype(str).str.strip()
    filtered_dataframe["text"] = filtered_dataframe["text"].astype(str).str.strip()
    filtered_dataframe = filtered_dataframe[
        (filtered_dataframe["author"] != "") & (filtered_dataframe["text"] != "")
    ].copy()

    author_counts = filtered_dataframe["author"].value_counts()
    valid_authors = author_counts[author_counts >= min_messages_per_author].index
    filtered_dataframe = filtered_dataframe[
        filtered_dataframe["author"].isin(valid_authors)
    ].copy()
    filtered_dataframe.reset_index(drop=True, inplace=True)

    if filtered_dataframe["author"].nunique() < 2:
        raise ValueError(
            "Sao necessarios pelo menos 2 autores com a quantidade minima de mensagens."
        )

    return filtered_dataframe


def _split_train_test(
    features_dataframe: pd.DataFrame,
    authors: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Tenta dividir com estratificacao e recua para split simples se necessario."""
    try:
        return train_test_split(
            features_dataframe,
            authors,
            test_size=test_size,
            random_state=random_state,
            stratify=authors,
        )
    except ValueError:
        return train_test_split(
            features_dataframe,
            authors,
            test_size=test_size,
            random_state=random_state,
        )


def train_authorship_models(
    messages_dataframe: pd.DataFrame,
    min_messages_per_author: int = 3,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, object]:
    """Treina e avalia tres modelos leves de identificacao de autoria."""
    filtered_dataframe = _filter_valid_messages(
        messages_dataframe,
        min_messages_per_author,
    )

    extracted_features = [
        extract_stylometric_features(text) for text in filtered_dataframe["text"]
    ]
    features_dataframe = pd.DataFrame(extracted_features)
    feature_names = features_dataframe.columns.tolist()
    authors = filtered_dataframe["author"]

    x_train, x_test, y_train, y_test = _split_train_test(
        features_dataframe,
        authors,
        test_size,
        random_state,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "gaussian_nb": GaussianNB(),
    }

    metrics: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        model.fit(x_train_scaled, y_train)
        predictions = model.predict(x_test_scaled)
        metrics[model_name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        }

    best_model_name = max(metrics, key=lambda name: metrics[name]["f1_macro"])

    return {
        "models": models,
        "metrics": metrics,
        "best_model_name": best_model_name,
        "best_model": models[best_model_name],
        "scaler": scaler,
        "feature_names": feature_names,
        "authors": sorted(authors.unique().tolist()),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "filtered_message_count": len(filtered_dataframe),
    }
