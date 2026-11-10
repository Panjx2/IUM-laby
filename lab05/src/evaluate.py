import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from data_prep import load_data, split_data

# katalog glowny projektu (jeden poziom nad src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Zadanie 4 - sciezka modelu i pliku z wynikami z pliku .env
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = BASE_DIR / os.environ["MODEL_PATH"]
RESULTS_PATH = BASE_DIR / os.environ["RESULTS_PATH"]


# Zadanie 3 - metryki zapisanego modelu na zbiorze testowym
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    return f"Accuracy: {accuracy:.4f}\n\n{report}"


# etap ewaluacji: python src/evaluate.py (bez ponownego trenowania)
if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit(f"Brak modelu {MODEL_PATH} - najpierw uruchom: python src/train.py")
    model = joblib.load(MODEL_PATH)
    # ten sam TEST_SIZE i RANDOM_SEED z .env daje ten sam zbior testowy co w train.py
    _, X_test, _, y_test = split_data(load_data())

    wyniki = (
        f"Model: {os.path.relpath(MODEL_PATH, BASE_DIR)}, "
        f"zbior testowy: {len(X_test)} probek\n"
        + evaluate(model, X_test, y_test)
    )
    print(wyniki)
    RESULTS_PATH.write_text(wyniki, encoding="utf-8")
    print(f"Wyniki zapisane do {os.path.relpath(RESULTS_PATH, BASE_DIR)}")
