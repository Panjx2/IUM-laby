import inspect
import time
from pathlib import Path

import numpy as np

import model_pytorch
import model_sklearn
from dane import podziel_dane, wczytaj_dane

# katalog glowny projektu (jeden poziom nad src/)
BASE_DIR = Path(__file__).resolve().parent.parent


def zmierz_czas(funkcja, *args, powtorzenia=1):
    # przy kilku powtorzeniach bierzemy najkrotszy czas (najmniej zaklocen od systemu)
    czasy = []
    for _ in range(powtorzenia):
        start = time.perf_counter()
        wynik = funkcja(*args)
        czasy.append(time.perf_counter() - start)
    return wynik, min(czasy)


def linie_kodu(*obiekty):
    # niepuste linie bez komentarzy potrzebne do zbudowania, trenowania i walidacji modelu
    zrodlo = "\n".join(inspect.getsource(o) for o in obiekty)
    return sum(1 for l in zrodlo.splitlines() if l.strip() and not l.strip().startswith("#"))


# Zadanie 6 - porownanie obu modeli
X, y, _ = wczytaj_dane()
X_train, X_test, y_train, y_test = podziel_dane(X, y)

# walidacja krzyzowa na tych samych foldach
cv_sk, czas_cv_sk = zmierz_czas(model_sklearn.walidacja_krzyzowa, X, y)
cv_pt, czas_cv_pt = zmierz_czas(model_pytorch.walidacja_krzyzowa, X, y)

# czas pojedynczego treningu na zbiorze treningowym z zadania 1
# (najlepszy z 5 pomiarow - pomija jednorazowa inicjalizacje PyTorcha)
_, czas_sk = zmierz_czas(model_sklearn.zbuduj_pipeline().fit, X_train, y_train, powtorzenia=5)
_, czas_pt = zmierz_czas(model_pytorch.trenuj, X_train, y_train, len(np.unique(y)), powtorzenia=5)

kod_sk = linie_kodu(model_sklearn.zbuduj_pipeline, model_sklearn.walidacja_krzyzowa)
kod_pt = linie_kodu(model_pytorch.Siec, model_pytorch.trenuj,
                    model_pytorch.przewiduj, model_pytorch.walidacja_krzyzowa)

wiersze = [
    ("scikit-learn", cv_sk, czas_sk, czas_cv_sk, kod_sk),
    ("PyTorch", cv_pt, czas_pt, czas_cv_pt, kod_pt),
]
tabela = [
    f"{'Model':<14}{'Acc srednia':>12}{'Odch. std':>11}{'Trening [s]':>13}"
    f"{'CV [s]':>9}{'Linie kodu':>12}"
]
for nazwa, cv, czas, czas_cv, kod in wiersze:
    tabela.append(
        f"{nazwa:<14}{cv.mean():>12.4f}{cv.std():>11.4f}{czas:>13.4f}"
        f"{czas_cv:>9.4f}{kod:>12}"
    )

foldy = [f"{'Fold':<14}" + "".join(f"{i:>8}" for i in range(1, len(cv_sk) + 1))]
for nazwa, cv, *_ in wiersze:
    foldy.append(f"{nazwa:<14}" + "".join(f"{acc:>8.4f}" for acc in cv))

lepszy, gorszy = sorted(wiersze, key=lambda w: w[1].mean(), reverse=True)
roznica = lepszy[1].mean() - gorszy[1].mean()
if roznica <= max(cv_sk.std(), cv_pt.std()):
    ocena_roznicy = "roznica miesci sie w odchyleniu std, wiec modele sa porownywalne"
else:
    ocena_roznicy = "roznica przekracza odchylenie std"
wniosek = "\n".join([
    "Wniosek:",
    f"- wyzsza srednia accuracy: {lepszy[0]} "
    f"({lepszy[1].mean():.4f} vs {gorszy[1].mean():.4f}),",
    f"  {ocena_roznicy},",
    f"- trening scikit-learn byl ~{czas_pt / czas_sk:.0f}x szybszy, a kod "
    f"{kod_pt / kod_sk:.1f}x krotszy ({kod_sk} vs {kod_pt} linii),",
    "- dla malego zbioru tabelarycznego lepszym wyborem jest scikit-learn;",
    "  PyTorch oplaca sie przy duzych danych nieustrukturyzowanych (obrazy, tekst).",
])

raport = "\n".join([
    "Walidacja krzyzowa (KFold, 5 foldow, accuracy)",
    *foldy,
    "",
    "Zestawienie",
    *tabela,
    "",
    wniosek,
])
print(raport)

with open(BASE_DIR / "wyniki.txt", "w", encoding="utf-8") as f:
    f.write(raport + "\n")
