from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dane import foldy, podziel_dane, wczytaj_dane


# Zadanie 2 - pipeline: skalowanie cech + klasyfikator
def zbuduj_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


# Zadanie 4 - 5-krotna walidacja krzyzowa (scaler dopasowywany osobno w kazdym foldzie)
def walidacja_krzyzowa(X, y):
    return cross_val_score(zbuduj_pipeline(), X, y, cv=foldy(), scoring="accuracy")


if __name__ == "__main__":
    X, y, _ = wczytaj_dane()
    X_train, X_test, y_train, y_test = podziel_dane(X, y)

    model = zbuduj_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model)
    print(f"\nAccuracy na zbiorze testowym: {accuracy_score(y_test, y_pred):.4f}")

    wyniki = walidacja_krzyzowa(X, y)
    print("\nWalidacja krzyzowa (KFold, 5 foldow):")
    for i, acc in enumerate(wyniki, start=1):
        print(f"  fold {i}: {acc:.4f}")
    print(f"Srednia: {wyniki.mean():.4f}, odchylenie std: {wyniki.std():.4f}")
