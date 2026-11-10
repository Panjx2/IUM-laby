import os
import signal
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Zadanie 1 - wiadomosc ze zmiennej srodowiskowej (z wartoscia domyslna)
GREETING = os.environ.get("GREETING", "Witaj z app.py (wartosc domyslna w kodzie)")
# katalog na wynik - w kontenerze /data, czyli miejsce montowania wolumenu
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
# opcjonalny port: jesli ustawiony, po zapisaniu wyniku aplikacja udostepnia /data
# po HTTP i dziala do zatrzymania (praca w tle, mapowanie portu w docker compose)
PORT = os.environ.get("PORT")


def zapisz_wynik(wiadomosc, katalog=DATA_DIR):
    katalog.mkdir(parents=True, exist_ok=True)
    plik = katalog / "wynik.txt"
    plik.write_text(wiadomosc + "\n", encoding="utf-8")
    return plik


if __name__ == "__main__":
    # flush=True - wyjscie od razu trafia do docker logs, bez czekania w buforze
    print(GREETING, flush=True)
    plik = zapisz_wynik(GREETING)
    print(f"Zapisano wiadomosc do {plik}", flush=True)

    if PORT:
        # python jako PID 1 ignoruje SIGTERM - bez tego docker stop czeka 10 s
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        handler = partial(SimpleHTTPRequestHandler, directory=str(DATA_DIR))
        print(f"Serwer HTTP z katalogiem {DATA_DIR} na porcie {PORT}", flush=True)
        HTTPServer(("", int(PORT)), handler).serve_forever()
