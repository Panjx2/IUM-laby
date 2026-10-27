import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dane import RANDOM_SEED, foldy, podziel_dane, wczytaj_dane

# siec i dane sa male - narzut wielowatkowosci jest wiekszy niz zysk
torch.set_num_threads(1)

# hiperparametry treningu
EPOKI = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001


# Zadanie 3 - siec w pelni polaczona z dwiema warstwami ukrytymi
class Siec(nn.Module):
    def __init__(self, n_cech, n_klas):
        super().__init__()
        self.warstwy = nn.Sequential(
            nn.Linear(n_cech, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_klas),
        )

    def forward(self, x):
        return self.warstwy(x)


def trenuj(X_train, y_train, n_klas, pokaz_strate=False):
    # skalowanie dopasowane tylko do danych treningowych (jak StandardScaler w Pipeline)
    scaler = StandardScaler().fit(X_train)
    X_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    # ziarno ustawiane przed kazdym treningiem: te same wagi startowe i kolejnosc batchy
    torch.manual_seed(RANDOM_SEED)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)
    model = Siec(X_t.shape[1], n_klas)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoka in range(1, EPOKI + 1):
        suma_straty = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)  # forward + obliczenie straty
            loss.backward()
            optimizer.step()
            suma_straty += loss.item() * len(xb)
        if pokaz_strate and epoka % 10 == 0:
            print(f"  epoka {epoka:2d}/{EPOKI}, strata: {suma_straty / len(X_t):.4f}")
    return model, scaler


def przewiduj(model, scaler, X):
    model.eval()
    with torch.no_grad():
        logity = model(torch.tensor(scaler.transform(X), dtype=torch.float32))
    return logity.argmax(dim=1).numpy()


# Zadanie 5 - reczna walidacja krzyzowa: nowa siec w kazdym foldzie
def walidacja_krzyzowa(X, y):
    n_klas = len(np.unique(y))
    wyniki = []
    for train_idx, test_idx in foldy().split(X):
        model, scaler = trenuj(X[train_idx], y[train_idx], n_klas)
        y_pred = przewiduj(model, scaler, X[test_idx])
        wyniki.append(accuracy_score(y[test_idx], y_pred))
    return np.array(wyniki)


if __name__ == "__main__":
    X, y, nazwy_klas = wczytaj_dane()
    X_train, X_test, y_train, y_test = podziel_dane(X, y)

    print(Siec(X.shape[1], len(nazwy_klas)))
    print(f"\nTrening ({EPOKI} epok, batch {BATCH_SIZE}, Adam lr={LEARNING_RATE}):")
    model, scaler = trenuj(X_train, y_train, len(nazwy_klas), pokaz_strate=True)
    y_pred = przewiduj(model, scaler, X_test)
    print(f"\nAccuracy na zbiorze testowym: {accuracy_score(y_test, y_pred):.4f}")

    wyniki = walidacja_krzyzowa(X, y)
    print("\nWalidacja krzyzowa (KFold, 5 foldow):")
    for i, acc in enumerate(wyniki, start=1):
        print(f"  fold {i}: {acc:.4f}")
    print(f"Srednia: {wyniki.mean():.4f}, odchylenie std: {wyniki.std():.4f}")
