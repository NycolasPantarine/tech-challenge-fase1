"""Definição da arquitetura da rede neural MLP para previsão de churn."""

import torch.nn as nn


class ChurnMLP(nn.Module):
    """MLP para classificação binária de churn.

    Arquitetura: input_size -> 32 -> 16 -> 1, com ReLU e Dropout.
    A saída é um logit (sem ativação) - aplique sigmoid para obter
    probabilidade.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)