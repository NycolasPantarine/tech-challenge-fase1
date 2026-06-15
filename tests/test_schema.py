"""Schema tests - validam o formato dos dados processados."""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

processed_data_schema = DataFrameSchema(
    {
        "customerID": Column(str, nullable=False),
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "tenure": Column(int, Check.ge(0)),
        "MonthlyCharges": Column(float, Check.ge(0)),
        "TotalCharges": Column(float, Check.ge(0)),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "Churn": Column(str, Check.isin(["Yes", "No"])),
    },
    strict=False,  # permite outras colunas além das listadas
)


def test_processed_data_matches_schema() -> None:
    """O dataset processado deve respeitar o schema esperado."""
    df = pd.read_csv("data/processed/telco_churn_clean.csv")

    processed_data_schema.validate(df)


def test_processed_data_has_no_missing_total_charges() -> None:
    """TotalCharges não deve ter valores nulos após o tratamento."""
    df = pd.read_csv("data/processed/telco_churn_clean.csv")

    assert df["TotalCharges"].isna().sum() == 0


def test_processed_data_churn_distribution() -> None:
    """A proporção de churn deve estar próxima de 26,5% (sanity check)."""
    df = pd.read_csv("data/processed/telco_churn_clean.csv")

    churn_rate = (df["Churn"] == "Yes").mean()

    assert 0.20 <= churn_rate <= 0.35
