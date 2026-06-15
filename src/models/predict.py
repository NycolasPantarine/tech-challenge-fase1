"""Funções para carregar artefatos treinados e realizar inferência."""

import logging
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.models.mlp import ChurnMLP

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MLP_INPUT_SIZE = 45


def load_preprocessor():
    """Carrega o ColumnTransformer treinado."""
    path = MODELS_DIR / "preprocessor.joblib"
    logger.info("Carregando preprocessor de %s", path)
    return joblib.load(path)


def load_logreg_pipeline():
    """Carrega o pipeline completo (preprocessor + Regressão Logística)."""
    path = MODELS_DIR / "logreg_pipeline.joblib"
    logger.info("Carregando pipeline de Regressão Logística de %s", path)
    return joblib.load(path)


def load_mlp_model() -> ChurnMLP:
    """Carrega a MLP treinada (arquitetura + pesos)."""
    path = MODELS_DIR / "mlp_model.pt"
    logger.info("Carregando pesos da MLP de %s", path)
    model = ChurnMLP(input_size=MLP_INPUT_SIZE)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def predict_churn_logreg(pipeline, data: pd.DataFrame) -> tuple[int, float]:
    """Realiza predição de churn usando o pipeline de Regressão Logística.

    Args:
        pipeline: Pipeline carregado via load_logreg_pipeline().
        data: DataFrame com uma linha contendo as features do cliente.

    Returns:
        Tupla (predicao, probabilidade), onde predicao é 0 ou 1.
    """
    proba = pipeline.predict_proba(data)[:, 1][0]
    prediction = int(proba >= 0.5)
    return prediction, float(proba)


def predict_churn_mlp(
    model: ChurnMLP, preprocessor, data: pd.DataFrame
) -> tuple[int, float]:
    """Realiza predição de churn usando a MLP.

    Args:
        model: ChurnMLP carregada via load_mlp_model().
        preprocessor: ColumnTransformer carregado via load_preprocessor().
        data: DataFrame com uma linha contendo as features do cliente.

    Returns:
        Tupla (predicao, probabilidade), onde predicao é 0 ou 1.
    """
    processed = preprocessor.transform(data)
    array = processed.toarray() if hasattr(processed, "toarray") else processed
    tensor = torch.tensor(array, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor)
        proba = torch.sigmoid(logits).item()

    prediction = int(proba >= 0.5)
    return prediction, proba