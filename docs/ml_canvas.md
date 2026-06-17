# ML Canvas — Previsão de Churn

## Problema de Negócio

**Quem usa o resultado?**
Time de retenção de clientes da operadora — usam a lista de clientes
em risco para priorizar ações proativas (ligações, ofertas, descontos).

**Qual a decisão que o modelo apoia?**
Quais clientes abordar primeiro numa campanha de retenção.

**O que acontece sem o modelo?**
A equipe age de forma reativa (após o cancelamento) ou aleatória
(sem priorização), desperdiçando recursos em clientes que não iriam
cancelar e perdendo clientes que iriam.

---

## Métricas

**Métrica técnica principal:** AUC-ROC
Escolhida porque o dataset é desbalanceado (73,5% / 26,5%) e a
acurácia seria enganosa. AUC-ROC mede o poder discriminativo do
modelo independente do threshold.

**Métricas secundárias:** F1-score, Precision e Recall da classe churn.

**Métrica de negócio:** custo de churn evitado.
- Falso negativo (cliente que cancela sem ser identificado) = perda
  do valor de vida do cliente (LTV)
- Falso positivo (cliente que não cancelaria mas recebe ação de
  retenção) = custo da ação (desconto, ligação)
- Em geral, custo do FN >> custo do FP → preferível ter recall alto

**SLO (Service Level Objective):**
- AUC-ROC ≥ 0.80 em produção
- Latência do endpoint `/predict` ≤ 500ms (p95)
- Disponibilidade da API ≥ 99%

---

## Dataset

**Fonte:** IBM Telco Customer Churn (público)
**Volume:** 7.043 clientes, 19 features
**Target:** `Churn` (Yes/No) → binário (1/0)
**Desbalanceamento:** 73,5% não-churn / 26,5% churn
**Data readiness:** dataset completo, sem missing values relevantes
(11 registros com TotalCharges vazio tratados)

---

## Features principais

| Feature | Tipo | Importância |
|---|---|---|
| `Contract` | Categórica | Alta — month-to-month tem 15x mais churn |
| `tenure` | Numérica | Alta — clientes novos cancelam muito mais |
| `MonthlyCharges` | Numérica | Média |
| `InternetService` | Categórica | Média |
| `PaymentMethod` | Categórica | Média |

---

## Modelos avaliados

| Modelo | AUC-ROC | Recomendação |
|---|---|---|
| DummyClassifier | 0.50 | Baseline de referência |
| Regressão Logística | 0.8421 | ✅ Recomendado para produção |
| Random Forest | 0.8185 | Descartado (pior que LogReg) |
| Gradient Boosting | 0.8433 | Alternativa viável |
| MLP (PyTorch) | 0.8419 | Exigência do Tech Challenge |

---

## Riscos e Limitações

- Recall de ~57%: 43% dos churns não são identificados
- Dataset estático: sem garantia de generalização temporal
- Features demográficas (gender, SeniorCitizen): risco de viés
- Sem dados comportamentais: frequência de uso, chamadas ao suporte

---

## Arquitetura de Deploy

**Modo:** Real-time (inferência por cliente, síncrona)
**Justificativa:** ações de retenção são individuais e sensíveis ao
tempo — batch processing introduziria latência desnecessária.
**API:** FastAPI + Uvicorn, endpoint `POST /predict`