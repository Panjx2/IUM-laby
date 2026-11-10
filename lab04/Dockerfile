# Zadanie 1 - obraz aplikacji app.py
FROM python:3.11-slim

WORKDIR /app

# Zadanie 2 - domyslna wartosc zmiennej, nadpisywana przez docker run -e GREETING=...
ENV GREETING="Witaj z Dockerfile (ENV)"

# caly kontekst budowania - co do niego trafia, okresla .dockerignore (zadanie 5)
COPY . .

CMD ["python", "app.py"]
