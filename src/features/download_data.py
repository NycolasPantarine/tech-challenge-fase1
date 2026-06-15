"""Script para download do dataset Telco Customer Churn."""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
OUTPUT_PATH = Path("data/raw/telco_customer_churn.csv")


def download_dataset(url: str, output_path: Path) -> None:
    """Baixa o dataset e salva localmente."""
    logger.info("Baixando dataset de %s", url)
    df = pd.read_csv(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset salvo em %s (%d linhas, %d colunas)", output_path, *df.shape)


if __name__ == "__main__":
    download_dataset(DATASET_URL, OUTPUT_PATH)