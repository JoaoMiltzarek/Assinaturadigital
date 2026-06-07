from __future__ import annotations

import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    pipeline_path = PROJECT_ROOT / "models" / "v1_pipeline.pkl"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            "models/v1_pipeline.pkl não encontrado. "
            "Rode primeiro: python scripts\\train_v1_pipeline.py"
        )

    pipeline = joblib.load(pipeline_path)

    examples = [
        "The universe is under no obligation to make sense to you.",
        "We need to act now for climate and future generations.",
        "This is amazing!! Check this out",
    ]

    predictions = pipeline.predict(examples)

    for text, author in zip(examples, predictions):
        print("-" * 80)
        print("Texto:", text)
        print("Autor previsto:", author)

    print("\nV1 pipeline ok")


if __name__ == "__main__":
    main()
