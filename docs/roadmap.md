# Roadmap do Projeto — Tech Challenge Fase 1

Checklist de progresso mapeado às 4 etapas do Tech Challenge. Atualizado ao
final de cada sessão de trabalho.

**Repositório:** https://github.com/NycolasPantarine/tech-challenge-fase1
**Prazo oficial:** 30/06/2026 | **Meta interna:** 26/06/2026

---

## Status Geral

| Bloco | Status | Sessão |
|---|---|---|
| 1. Fundação (EDA + Baselines) | ✅ Concluído | 14/06 |
| 2. Modelagem (MLP PyTorch) | ✅ Concluído | 14/06 |
| 3. Engenharia (src/, API, testes) | ✅ Concluído | 15/06 |
| 4. Fechamento (Model Card, README, vídeo) | 🔶 Em andamento | 17/06 |

---

## Bloco 1 — Fundação ✅ (14/06)

- [x] Estrutura do repositório criada (src/, data/, models/, tests/, notebooks/, docs/)
- [x] Ambiente configurado com `uv` (pyproject.toml + uv.lock)
- [x] Dataset Telco Customer Churn baixado (`src/features/download_data.py`)
- [x] EDA completa (`notebooks/01_eda.ipynb`)
  - [x] Tratamento de `TotalCharges` (11 valores vazios → 0, tenure=0)
  - [x] Análise de desbalanceamento de classes (73,5% / 26,5%)
  - [x] Insight: tenure baixo → churn alto
  - [x] Insight: contrato month-to-month → churn 15x maior
- [x] Dataset limpo salvo em `data/processed/telco_churn_clean.csv`
- [x] Baseline DummyClassifier (AUC=0.50) — `notebooks/02_baselines.ipynb`
- [x] Baseline Regressão Logística (AUC=0.84)
- [x] Pipeline de pré-processamento (ColumnTransformer: StandardScaler + OneHotEncoder)
- [x] Experimentos registrados no MLflow (`mlflow.db` na raiz, experiment `churn-prediction-baselines`)
- [x] `docs/decisoes_tecnicas.md` criado e documentado (seções 1-7)

---

## Bloco 2 — Modelagem (MLP) ✅ (14/06)

- [x] MLP construída em PyTorch (`notebooks/03_mlp_pytorch.ipynb`)
  - Arquitetura: 45 → 32 → 16 → 1, ReLU, Dropout(0.2)
  - Loss: BCEWithLogitsLoss | Optimizer: Adam (lr=0.001)
- [x] Loop de treinamento com early stopping (patience=10, batch=32)
- [x] Avaliação no conjunto de teste (AUC=0.8419)
- [x] Comparação MLP vs. baselines (≥4 métricas) — documentada
- [x] Análise crítica: MLP ≈ Regressão Logística (recomendação: modelo mais simples)
- [x] Análise de trade-off custo (falso positivo vs. falso negativo)
- [x] Experimento MLP registrado no MLflow
- [x] Artefatos persistidos em `models/`:
  - `preprocessor.joblib`
  - `logreg_pipeline.joblib` (pipeline completo, recomendado para produção)
  - `mlp_model.pt` (state_dict, input_size=45)
- [x] `docs/decisoes_tecnicas.md` atualizado (seções 8-10)

---

## Bloco 3 — Engenharia ✅ (15/06)

Referência: Etapa 3 do Tech Challenge (Eng. Software Aulas 01-05, Bibliotecas
Aula 02, APIs Aulas 01-04)

- [x] Refatorar lógica dos notebooks em módulos `src/`
  - [x] `src/features/preprocessing.py` — build_preprocessor, split_features_target
  - [x] `src/models/mlp.py` — classe ChurnMLP
  - [x] `src/models/predict.py` — load/predict para logreg e mlp
  - [x] `src/api/schemas.py` — CustomerFeatures, PredictionResponse, HealthResponse
  - [x] `src/api/main.py` — FastAPI com lifespan, middleware, /health, /predict
- [x] API FastAPI funcional e testada:
  - [x] `/health` → `{"status": "ok", "models_loaded": true}`
  - [x] `/predict` → predição com logreg (padrão) ou mlp via query param
  - [x] Validação Pydantic com Literal types (422 em input inválido)
  - [x] Logging estruturado (sem print())
  - [x] Middleware de latência
- [x] Testes automatizados — 12 testes, todos passando:
  - [x] `tests/test_smoke.py` — 4 testes (preprocessor, mlp, split, artefatos)
  - [x] `tests/test_schema.py` — 3 testes (schema, nulos, distribuição churn)
  - [x] `tests/test_api.py` — 5 testes (health, predict logreg/mlp, erros 400/422)
- [x] `ruff` configurado e sem erros (`make lint` → All checks passed!)
- [x] Makefile funcional: `make install`, `make lint`, `make test`, `make run`

## Bloco 4 — Fechamento 🔶 (17/06)

- [x] Model Card completo (`docs/model_card.md`)
- [x] ML Canvas (`docs/ml_canvas.md`)
- [x] Plano de monitoramento (dentro do Model Card)
- [x] Decisão de arquitetura de deploy (real-time + justificativa)
- [x] README final completo com setup, execução e URL de produção
- [x] Deploy em nuvem — GCP Cloud Run
      URL: https://churn-api-311138300643.us-central1.run.app
- [ ] Roteiro do vídeo STAR
- [ ] Gravação do vídeo (5 min, método STAR)