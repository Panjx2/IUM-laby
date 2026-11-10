# Zadanie 4 - druga usluga: serwer HTTP z katalogiem wynikow treningu
FROM python:3.11-slim

WORKDIR /srv

# print() od razu trafia do logow kontenera, bez czekania w buforze
ENV PYTHONUNBUFFERED=1

# sama biblioteka standardowa - ten obraz nie potrzebuje bibliotek ML z uslugi train
COPY serwer.py .

# tylko dokumentuje port - na hosta wystawia go sekcja ports w docker-compose.yml
EXPOSE 8080

CMD ["python", "serwer.py"]
