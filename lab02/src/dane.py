import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, train_test_split

# wspolne ustawienia dla obu modeli
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_FOLDOW = 5


# Zadanie 1 - wspolny zbior danych: wine (178 probek, 13 cech, 3 klasy)
def wczytaj_dane():
    wine = load_wine()
    return wine.data, wine.target, wine.target_names


def podziel_dane(X, y):
    # podzial 80/20 ze stratyfikacja - oba modele dostaja dokladnie te same zbiory
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )


def foldy():
    # ten sam podzial na foldy dla walidacji krzyzowej obu modeli
    return KFold(n_splits=N_FOLDOW, shuffle=True, random_state=RANDOM_SEED)


if __name__ == "__main__":
    X, y, nazwy_klas = wczytaj_dane()
    X_train, X_test, y_train, y_test = podziel_dane(X, y)

    print("Zbior danych: wine")
    print(f"Rozmiar X: {X.shape}, liczba klas: {len(nazwy_klas)}")
    print("Liczebnosc klas:", dict(zip(nazwy_klas.tolist(), np.bincount(y).tolist())))
    print(f"\nZbior treningowy: {X_train.shape}, testowy: {X_test.shape}")
    print("Klasy w zbiorze treningowym:", np.bincount(y_train).tolist())
    print("Klasy w zbiorze testowym:   ", np.bincount(y_test).tolist())
