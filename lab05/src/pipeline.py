"""Uruchamia po kolei etapy pipeline'u z lab. 3, kazdy jak lokalnie: python src/<etap>.

W trakcie pracy udostepnia swoj stan pod /status, zeby inne uslugi z docker-compose.yml
mogly sprawdzic po nazwie uslugi (http://train:5000/status), jaki etap wlasnie trwa.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SRC = Path(__file__).resolve().parent
ETAPY = ["data_prep.py", "train.py", "evaluate.py"]
STATUS_PORT = int(os.environ.get("STATUS_PORT", "5000"))

# stan pipeline'u zwracany pod /status
stan = {"etap": None, "ukonczone": []}


class Status(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/status":
            self.send_error(404)
            return
        tresc = json.dumps(stan).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(tresc)))
        self.end_headers()
        self.wfile.write(tresc)


if __name__ == "__main__":
    # serwer w watku-demonie konczy sie razem z pipeline'em - kontener sie zatrzymuje
    serwer = HTTPServer(("", STATUS_PORT), Status)
    threading.Thread(target=serwer.serve_forever, daemon=True).start()

    for etap in ETAPY:
        stan["etap"] = etap
        print(f"=== etap: {etap} ===", flush=True)
        kod = subprocess.run([sys.executable, str(SRC / etap)]).returncode
        if kod != 0:
            # blad etapu przerywa pipeline i konczy kontener tym samym kodem
            sys.exit(kod)
        stan["ukonczone"].append(etap)
