"""Smoke tests - verificam que os componentes essenciais carregam sem erro."""

import pandas as pd

from src.features.preprocessing import build_preprocessor, split_features_target
from src.models.mlp import ChurnMLP
from src.models.predict import (
    load_logreg_pipeline,
    load_mlp_model,
    load_preprocessor,
)


def test_preprocessor_builds() -> None:
    """O ColumnTransformer deve ser construído sem erro."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_mlp_instantiates() -> None:
    """A classe ChurnMLP deve instanciar e processar um tensor de exemplo."""
    import torch

    model = ChurnMLP(input_size=45)
    dummy_input = torch.zeros((1, 45))
    output = model(dummy_input)

    assert output.shape == (1, 1)


def test_split_features_target() -> None:
    """split_features_target deve separar X e y corretamente."""
    df = pd.DataFrame(
        {
            "customerID": ["0001"],
            "Churn": ["Yes"],
            "tenure": [5],
        }
    )

    X, y = split_features_target(df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert y.iloc[0] == 1


def test_artifacts_load() -> None:
    """Os artefatos salvos em models/ devem carregar sem erro."""
    preprocessor = load_preprocessor()
    logreg = load_logreg_pipeline()
    mlp = load_mlp_model()

    assert preprocessor is not None
    assert logreg is not None
    assert mlp is not None