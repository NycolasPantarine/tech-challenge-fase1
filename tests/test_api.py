"""API tests - testam os endpoints da aplicação FastAPI."""

from fastapi.testclient import TestClient

from src.api.main import app, load_models

# Carrega os modelos manualmente, já que o TestClient não dispara
# o evento "startup" automaticamente em todas as versões do FastAPI.
load_models()

client = TestClient(app)

VALID_CUSTOMER = {
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
    "TotalCharges": 29.85,
}


def test_health_endpoint() -> None:
    """O endpoint /health deve retornar status ok e modelos carregados."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


def test_predict_with_logreg() -> None:
    """O endpoint /predict deve retornar uma predição válida com logreg."""
    response = client.post("/predict?model=logreg", json=VALID_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_used"] == "logreg"


def test_predict_with_mlp() -> None:
    """O endpoint /predict deve retornar uma predição válida com mlp."""
    response = client.post("/predict?model=mlp", json=VALID_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_used"] == "mlp"


def test_predict_with_invalid_model_param() -> None:
    """Um valor inválido para 'model' deve retornar erro 400."""
    response = client.post("/predict?model=invalid", json=VALID_CUSTOMER)

    assert response.status_code == 400


def test_predict_with_invalid_gender() -> None:
    """Um valor inválido para 'gender' deve retornar erro 422 (Pydantic)."""
    invalid_customer = {**VALID_CUSTOMER, "gender": "Invalido"}

    response = client.post("/predict?model=logreg", json=invalid_customer)

    assert response.status_code == 422
    