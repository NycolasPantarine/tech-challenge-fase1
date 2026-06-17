# Churn Prediction — Tech Challenge Fase 1

Projeto de previsão de churn para uma operadora de telecomunicações,
desenvolvido como Tech Challenge da pós-graduação em Machine Learning
Engineering (FIAP).

O objetivo é identificar clientes com risco de cancelamento usando uma
pipeline completa: da exploração dos dados até uma API de inferência em
produção.

---

## O que foi construído

- **EDA** do dataset Telco Customer Churn (IBM) com análise de
  desbalanceamento, distribuição de features e insights de negócio
- **Baselines** com DummyClassifier e Regressão Logística (scikit-learn)
- **MLP** treinada com PyTorch, com early stopping e comparação contra
  os baselines
- **Tracking de experimentos** com MLflow
- **API de inferência** com FastAPI, validação Pydantic e logging
  estruturado
- **12 testes automatizados** (smoke, schema, API) com pytest
- **Linting** com ruff, sem erros

---

## Resultados

| Modelo | AUC-ROC | F1 (churn) |
|---|---|---|
| DummyClassifier | 0.50 | 0.00 |
| Regressão Logística | **0.84** | 0.60 |
| MLP (PyTorch) | 0.84 | 0.60 |

A MLP e a Regressão Logística chegaram ao mesmo resultado. Para esse
dataset tabular, a complexidade extra da rede neural não trouxe ganho
mensurável. O modelo recomendado para produção é a Regressão Logística.

---

## Estrutura do projeto

tech-challenge-fase1/

├── data/

│   ├── raw/                  # dataset original (não versionado)

│   └── processed/            # dataset limpo

├── docs/

│   ├── model_card.md         # documentação do modelo

│   ├── decisoes_tecnicas.md  # diário de decisões técnicas

│   └── roadmap.md            # progresso do projeto

├── models/                   # artefatos treinados

│   ├── preprocessor.joblib

│   ├── logreg_pipeline.joblib

│   └── mlp_model.pt

├── notebooks/                # exploração e experimentos

│   ├── 01_eda.ipynb

│   ├── 02_baselines.ipynb

│   └── 03_mlp_pytorch.ipynb

├── src/

│   ├── features/             # pré-processamento

│   ├── models/               # arquitetura e inferência

│   └── api/                  # FastAPI

├── tests/                    # testes automatizados

├── Makefile

└── pyproject.toml

---

## Como rodar

### Pré-requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) instalado

### Instalação

```bash
git clone https://github.com/NycolasPantarine/tech-challenge-fase1.git
cd tech-challenge-fase1
make install
```

### Baixar o dataset

```bash
python src/features/download_data.py
```

### Rodar os testes

```bash
make test
```

### Verificar qualidade do código

```bash
make lint
```

### Subir a API

```bash
make run
```

A API fica disponível em `http://127.0.0.1:8000`.
Documentação interativa: `http://127.0.0.1:8000/docs`.

### Exemplo de requisição

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

Resposta esperada:

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.6137,
  "model_used": "logreg"
}
```

### Visualizar experimentos no MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acessa `http://127.0.0.1:5000`.

---

## Bibliotecas principais

| Biblioteca | Uso |
|---|---|
| PyTorch | Construção e treinamento da MLP |
| scikit-learn | Baselines, pipeline de pré-processamento |
| MLflow | Tracking de experimentos |
| FastAPI | API de inferência |
| Pydantic | Validação de dados de entrada |
| pandera | Validação de schema do dataset |
| pytest | Testes automatizados |
| ruff | Linting |

## Deploy em produção

A API está disponível publicamente no Google Cloud Run:

**Base URL:** `https://churn-api-311138300643.us-central1.run.app`

| Endpoint | Descrição |
|---|---|
| `GET /health` | Status da API |
| `POST /predict` | Predição de churn |
| `GET /docs` | Documentação interativa |

---

## Documentação adicional

- [Model Card](docs/model_card.md) — performance, limitações e vieses
- [Decisões Técnicas](docs/decisoes_tecnicas.md) — raciocínio por trás
  de cada escolha do projeto