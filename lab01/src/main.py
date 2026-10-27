import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# katalog glowny projektu (jeden poziom nad src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Zadanie 6 - wczytanie konfiguracji z pliku .env
load_dotenv(BASE_DIR / ".env")
data_path = BASE_DIR / os.getenv("DATA_PATH", "data/iris.csv")
test_size = float(os.getenv("TEST_SIZE", 0.2))
random_seed = int(os.getenv("RANDOM_SEED", 42))


# Zadanie 3 - wczytanie zbioru iris i zamiana na DataFrame
def wczytaj_dane(sciezka):
    # jesli pliku z danymi jeszcze nie ma, zapisujemy go z gotowego zbioru sklearn
    if not sciezka.exists():
        iris = load_iris(as_frame=True)
        df = iris.frame
        df["target"] = df["target"].map(dict(enumerate(iris.target_names)))
        sciezka.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(sciezka, index=False)
    return pd.read_csv(sciezka)


df = wczytaj_dane(data_path)

print("Pierwsze wiersze:")
print(df.head())
print("\nRozmiar zbioru:", df.shape)
print("\nStatystyki opisowe:")
print(df.describe())

# Zadanie 4 - podzial na zbior treningowy i testowy oraz trening modelu
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed, stratify=y
)
print(f"\nZbior treningowy: {len(X_train)}, testowy: {len(X_test)}")

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Zadanie 5 - ocena modelu na zbiorze testowym
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
raport = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(raport)

# zapis wynikow do pliku
with open(BASE_DIR / "wyniki.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(raport)
