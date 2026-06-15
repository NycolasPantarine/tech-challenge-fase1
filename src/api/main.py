"""API FastAPI para predição de churn de clientes."""

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.models.predict import (
    load_logreg_pipeline,
    load_mlp_model,
    load_preprocessor,
    predict_churn_logreg,
    predict_churn_mlp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Artefatos carregados uma vez, na inicialização
_artifacts: dict = {}


def load_models() -> None:
    """Carrega os modelos e o preprocessor na inicialização da API."""
    logger.info("Carregando artefatos de modelo...")
    _artifacts["preprocessor"] = load_preprocessor()
    _artifacts["logreg_pipeline"] = load_logreg_pipeline()
    _artifacts["mlp_model"] = load_mlp_model()
    logger.info("Artefatos carregados com sucesso.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação: carrega modelos no startup."""
    load_models()
    yield
    _artifacts.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="API para predição de churn de clientes de telecomunicações",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_request_latency(request: Request, call_next):
    """Middleware que registra o tempo de resposta de cada requisição."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "%s %s - status=%d - %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Verifica o status da API e se os modelos estão carregados."""
    models_loaded = all(
        key in _artifacts for key in ("preprocessor", "logreg_pipeline", "mlp_model")
    )
    return HealthResponse(status="ok", models_loaded=models_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, model: str = "logreg") -> PredictionResponse:
    """Realiza a predição de churn para um cliente.

    Args:
        customer: Features do cliente.
        model: Qual modelo usar - "logreg" (padrão, recomendado) ou "mlp".

    Returns:
        Predição (0 ou 1) e probabilidade de churn.
    """
    if model not in ("logreg", "mlp"):
        raise HTTPException(
            status_code=400, detail="Parâmetro 'model' deve ser 'logreg' ou 'mlp'."
        )

    data = pd.DataFrame([customer.model_dump()])

    if model == "logreg":
        prediction, probability = predict_churn_logreg(
            _artifacts["logreg_pipeline"], data
        )
    else:
        prediction, probability = predict_churn_mlp(
            _artifacts["mlp_model"], _artifacts["preprocessor"], data
        )

        logger.info(
        "Predição realizada: model=%s, churn=%d, proba=%.4f",
        model, prediction, probability,
        )

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        model_used=model,
    )
