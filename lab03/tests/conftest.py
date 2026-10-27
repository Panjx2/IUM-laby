import os

# data_prep.py czyta parametry z .env juz przy imporcie, a .env nie trafia do repozytorium
# (np. w CI go nie ma) - testy ustawiaja wlasne wartosci, jesli zmienne nie sa ustawione
os.environ.setdefault("DATA_PATH", "data/wine.csv")
os.environ.setdefault("TEST_SIZE", "0.2")
os.environ.setdefault("RANDOM_SEED", "42")
