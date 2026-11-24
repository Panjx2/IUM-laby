# IUM – laboratoria

[![CI](https://github.com/Panjx2/IUM-laby/actions/workflows/ci.yml/badge.svg)](https://github.com/Panjx2/IUM-laby/actions/workflows/ci.yml)

Laboratoria z przedmiotu Inżynieria uczenia maszynowego.

| Gdzie | Temat |
|---|---|
| `lab01/` | środowisko pracy, eksploracja danych i model klasyfikacji |
| `lab02/` | pipeline scikit-learn vs PyTorch, walidacja krzyżowa i porównanie modeli |
| `lab03/` | podział pipeline'u na moduły, konfiguracja `.env`, testy pytest |
| `lab04/` | podstawy Dockera: Dockerfile, zmienne środowiskowe, wolumen, docker compose |
| `lab05/` | konteneryzacja pipeline'u ML: obraz z zależnościami, Compose, limity zasobów |
| `.github/workflows/ci.yml` | lab. 6: ciągła integracja w GitHub Actions dla projektu z `lab05/` |

## CI

Odznaka pokazuje wynik ostatniego przebiegu workflow `CI` na gałęzi `main`.
Przy każdym pushu do `main` i w każdym pull requeście workflow uruchamia w kontenerze
`python:3.11-slim` testy `pytest` projektu z `lab05/`, a gdy przejdą, trenuje model
i zapisuje go razem z metrykami jako artefakt przebiegu.
