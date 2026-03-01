import json
import os
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parents[1]  # backend/app
DATA_PATH = BASE / "data" / "nlu_train.jsonl"
OUT_PATH  = BASE.parents[0] / "models" / "intent_gate.joblib"  # backend/models

def load_jsonl(path: Path):
    X, y = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            X.append(obj["text"])
            y.append(obj["intent"])
    return X, y

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe {DATA_PATH}. Crea el dataset primero.")

    X, y = load_jsonl(DATA_PATH)

    # split simple para ver métricas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    # Pipeline clásico y muy sólido
    clf = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1,2),
                lowercase=True
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3,5),
                lowercase=True
            ))
        ])),
        ("lr", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_PATH)
    print(f"Modelo guardado en: {OUT_PATH}")

if __name__ == "__main__":
    main()