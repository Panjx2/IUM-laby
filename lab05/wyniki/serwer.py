"""Usluga wyniki: serwer HTTP z katalogiem wynikow treningu (model i metryki).

Zadanie 6 - przed startem serwera odpytuje usluge train po nazwie z docker-compose.yml
i czeka na koniec pipeline'u, zamiast polegac na samym depends_on.
"""
import json
import os
import signal
import socket
import sys
import time
import urllib.request
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit

TRAIN_URL = os.environ.get("TRAIN_URL", "http://train:5000/status")
HOST = urlsplit(TRAIN_URL).hostname
KATALOG = Path(os.environ.get("KATALOG", "/wyniki"))
PORT = int(os.environ.get("PORT", "8080"))
PROBY = 10  # ile razy (co sekunde) czekac na pierwsza odpowiedz uslugi train


def stan_train():
    """Zwraca (adres IP, stan z /status) uslugi train albo None, gdy nie odpowiada."""
    try:
        # nazwe uslugi na adres kontenera zamienia DNS sieci docker compose
        ip = socket.gethostbyname(HOST)
        with urllib.request.urlopen(TRAIN_URL, timeout=2) as odpowiedz:
            return ip, json.load(odpowiedz)
    except (OSError, ValueError):
        return None


def czekaj_na_train():
    print(f"Sprawdzam usluge train: {TRAIN_URL}")
    # depends_on gwarantuje tylko start kontenera train, a nie to, ze juz odpowiada
    for proba in range(1, PROBY + 1):
        odpowiedz = stan_train()
        if odpowiedz:
            break
        print(f"{HOST} nie odpowiada (proba {proba}/{PROBY})")
        time.sleep(1)
    else:
        print(f"{HOST} niedostepny - pomijam czekanie na trening")
        return

    etap = None
    while odpowiedz:
        ip, stan = odpowiedz
        if stan["etap"] != etap:
            etap = stan["etap"]
            ukonczone = ", ".join(stan["ukonczone"]) or "-"
            print(f"{HOST} ({ip}) odpowiada - etap: {etap}, ukonczone: {ukonczone}")
        time.sleep(1)
        odpowiedz = stan_train()
    # train odpowiadal i przestal: pipeline sie zakonczyl, a jego kontener zatrzymal
    print(f"{HOST} przestal odpowiadac - pipeline zakonczony, kontener zatrzymany")


if __name__ == "__main__":
    # python jako PID 1 ignoruje SIGTERM - bez tego docker compose down czeka 10 s
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    czekaj_na_train()
    # wspolny wolumen: to, co zapisal train, jest widoczne w tym kontenerze
    pliki = sorted(p.name for p in KATALOG.iterdir() if not p.name.startswith("."))
    if not pliki:
        sys.exit(f"Brak wynikow treningu w {KATALOG}")
    print(f"Wyniki na wolumenie {KATALOG}: {', '.join(pliki)}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(KATALOG))
    print(f"Serwer HTTP z katalogiem {KATALOG} na porcie {PORT}")
    HTTPServer(("", PORT), handler).serve_forever()
