import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# katalog glowny projektu (jeden poziom nad src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Zadanie 4 - parametry danych z pliku .env (brak klucza = KeyError, bez wartosci domyslnych)
load_dotenv(BASE_DIR / ".env")
DATA_PATH = BASE_DIR / os.environ["DATA_PATH"]
TEST_SIZE = float(os.environ["TEST_SIZE"])
RANDOM_SEED = int(os.environ["RANDOM_SEED"])

TARGET = "target"


# Zadanie 1 - wczytanie danych z CSV do DataFrame
def load_data(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Brak pliku {path} - najpierw uruchom: python src/data_prep.py"
        )
    return pd.read_csv(path)


# Zadanie 1 - podzial na zbior treningowy i testowy (ze stratyfikacja, jak w lab. 2)
def split_data(df, test_size=TEST_SIZE, seed=RANDOM_SEED):
    if df.empty:
        raise ValueError("Pusty zbior danych")
    if TARGET not in df.columns:
        raise ValueError(f"Brak kolumny docelowej '{TARGET}'")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size musi byc z przedzialu (0, 1), podano {test_size}")
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def save_wine(path):
    # zbior wine z lab. 2 (178 probek, 13 cech, 3 klasy) zapisany jako CSV
    wine = load_wine(as_frame=True)
    df = wine.frame
    df[TARGET] = df[TARGET].map(dict(enumerate(wine.target_names)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# etap przygotowania danych: python src/data_prep.py
if __name__ == "__main__":
    if not DATA_PATH.exists():
        save_wine(DATA_PATH)
        print(f"Zapisano zbior wine do {os.path.relpath(DATA_PATH, BASE_DIR)}")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Dane: {os.path.relpath(DATA_PATH, BASE_DIR)}, rozmiar: {df.shape}")
    print(f"Parametry: TEST_SIZE={TEST_SIZE}, RANDOM_SEED={RANDOM_SEED}")
    print(f"Zbior treningowy: {X_train.shape}, testowy: {X_test.shape}")
    print("Klasy w zbiorze treningowym:", y_train.value_counts().sort_index().to_dict())
    print("Klasy w zbiorze testowym:   ", y_test.value_counts().sort_index().to_dict())
