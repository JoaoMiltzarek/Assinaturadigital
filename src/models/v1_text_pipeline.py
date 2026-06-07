from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from src.features_v1 import FEATURES_V1, extrair_features_v1


class ManualStylometryTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]):
        rows = []

        for text in X:
            features = extrair_features_v1(text)
            rows.append([features[name] for name in FEATURES_V1])

        matrix = np.asarray(rows, dtype=float)
        return csr_matrix(matrix)


def build_v1_pipeline(classifier) -> Pipeline:
    manual_features_pipeline = Pipeline(
        steps=[
            ("manual_features", ManualStylometryTransformer()),
            ("manual_scaler", StandardScaler(with_mean=False)),
        ]
    )

    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=6000,
        lowercase=True,
    )

    combined_features = FeatureUnion(
        transformer_list=[
            ("manual", manual_features_pipeline),
            ("char_tfidf", char_tfidf),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("features", combined_features),
            ("classifier", classifier),
        ]
    )

    return pipeline
