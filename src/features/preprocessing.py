"""Construção do pipeline de pré-processamento de features."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def build_preprocessor() -> ColumnTransformer:
    """Cria o ColumnTransformer para features numéricas e categóricas.

    Returns:
        ColumnTransformer não treinado, pronto para fit/transform.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) do target (y) a partir do dataframe limpo.

    Args:
        df: DataFrame com a coluna 'Churn' ('Yes'/'No') e 'customerID'.

    Returns:
        Tupla (X, y) onde y é binário (1 = churn, 0 = não-churn).
    """
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})
    return X, y