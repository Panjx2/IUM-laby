# Zadanie 4 - serwer HTTP ze statycznym plikiem index.html
FROM python:3.11-slim

WORKDIR /www

COPY index.html .

# tylko dokumentuje port - na hosta wystawia go dopiero docker run -p
EXPOSE 8000

CMD ["python", "-m", "http.server", "8000"]
