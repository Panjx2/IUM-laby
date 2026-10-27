import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_data, split_data

# katalog glowny projektu (jeden poziom nad src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Zadanie 4 - hiperparametry modelu i sciezka zapisu z pliku .env
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = BASE_DIR / os.environ["MODEL_PATH"]
MODEL_C = float(os.environ["MODEL_C"])
MODEL_MAX_ITER = int(os.environ["MODEL_MAX_ITER"])


# Zadanie 2 - pipeline z lab. 2: skalowanie cech + regresja logistyczna
def build_model(c=MODEL_C, max_iter=MODEL_MAX_ITER):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=max_iter)),
    ])


def save_model(model, path=MODEL_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


# etap treningu: python src/train.py
if __name__ == "__main__":
    X_train, _, y_train, _ = split_data(load_data())
    model = build_model().fit(X_train, y_train)
    save_model(model)

    print(model)
    print(f"\nTrening na {len(X_train)} probkach, "
          f"accuracy treningowe: {model.score(X_train, y_train):.4f}")
    print(f"Model zapisany do {os.path.relpath(MODEL_PATH, BASE_DIR)}")
