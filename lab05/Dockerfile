# Zadanie 1 - obraz pipeline'u z lab. 3
FROM python:3.11-slim

WORKDIR /app

# print() od razu trafia do logow kontenera, bez czekania w buforze
ENV PYTHONUNBUFFERED=1

# najpierw same zaleznosci: ta warstwa zostaje w cache, dopoki requirements.txt
# sie nie zmieni - zmiana w kodzie nie powoduje ponownej instalacji bibliotek
COPY requirements.txt .
# uv zamiast pip: te same pakiety, szybsza instalacja. Program uv jest montowany
# z oficjalnego obrazu tylko na czas tego kroku, wiec nie trafia do gotowego obrazu
RUN --mount=from=ghcr.io/astral-sh/uv:0.11.2,source=/uv,target=/bin/uv \
    uv pip install --system --no-cache --compile-bytecode -r requirements.txt

# src/ i .env.example - reszte kontekstu budowania odcina .dockerignore (zadanie 5)
COPY . .

CMD ["python", "src/train.py"]
