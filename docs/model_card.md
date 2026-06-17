# Model Card — Previsão de Churn

## O que é esse modelo

Modelo de classificação binária que prevê se um cliente de uma operadora de
telecomunicações vai cancelar o serviço (churn). O objetivo é antecipar
cancelamentos para que a empresa possa agir antes que aconteçam.

Desenvolvido como projeto acadêmico (Tech Challenge — Pós-graduação em
Machine Learning Engineering, FIAP), usando o dataset público Telco Customer
Churn da IBM.

---

## Dataset

Dados históricos de 7.043 clientes de uma operadora fictícia, com 19
features por cliente (plano contratado, tempo de contrato, serviços
adicionais, forma de pagamento, etc.).

O dataset tem desbalanceamento natural: 73,5% dos clientes não cancelaram,
26,5% cancelaram. Isso significa que acurácia não é uma boa métrica aqui —
um modelo que prevê "não cancela" para todo mundo já acerta 73,5% das vezes
sem aprender nada.

**Limpeza aplicada:** 11 clientes tinham `TotalCharges` em branco (todos com
`tenure=0`, ou seja, ainda não tinham recebido nenhuma cobrança). Esses
valores foram preenchidos com `0`, que é o valor correto para quem ainda não
foi cobrado.

---

## Como foi treinado

Pipeline de pré-processamento com `scikit-learn`:
- Features numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`,
  `SeniorCitizen`) → normalizadas com `StandardScaler`
- Features categóricas (15 colunas) → `OneHotEncoder`, resultando em 45
  features no total

Três modelos treinados e comparados:

**DummyClassifier** — baseline de referência. Sempre prevê "não cancela".
AUC-ROC de 0.50 — equivalente a jogar uma moeda.

**Regressão Logística** — primeiro modelo real. AUC-ROC de 0.84 com uma
linha de código. Esse resultado já é bom e serve como baseline competitivo.

**MLP (PyTorch)** — rede neural com arquitetura `45→32→16→1`, ReLU e
Dropout(0.2). Treinada com Adam (lr=0.001), batch de 32, early stopping com
patience=10. Convergiu na época 6, parou na época 16.

---

## Resultados

| Modelo | AUC-ROC | F1 (churn) | Recall (churn) | Precision (churn) |
|---|---|---|---|---|
| Dummy | 0.50 | 0.00 | 0.00 | 0.00 |
| Regressão Logística | **0.8421** | 0.60 | 0.56 | 0.66 |
| MLP | 0.8419 | 0.60 | 0.58 | 0.62 |

A MLP e a Regressão Logística chegaram ao mesmo resultado na prática
(diferença de AUC de 0.0002). Para esse dataset tabular de ~7k registros,
a complexidade extra da rede neural não trouxe ganho mensurável.

**Modelo recomendado para produção: Regressão Logística.** Mesma
performance, mais simples de manter, mais fácil de interpretar.

---

## O que o modelo erra

Com recall de ~0.57, o modelo deixa passar ~43% dos clientes que vão
cancelar sem identificar. Isso é um limite real — não deve ser usado como
única fonte de decisão.

Os principais pontos cegos:

- **Clientes recém-assinados** (`tenure=0`): o modelo viu poucos exemplos
  desse grupo no treino e pode ser menos confiável para eles.
- **Features que não temos:** o modelo só vê dados contratuais e
  demográficos. Dados comportamentais (frequência de uso, chamadas ao
  suporte, atrasos de pagamento) provavelmente melhorariam muito a
  performance — mas não estavam disponíveis neste dataset.
- **Generalização:** foi treinado em um dataset público simplificado.
  Numa operadora real, com dados reais, o comportamento pode ser diferente.

---

## Vieses conhecidos

O modelo usa `gender` e `SeniorCitizen` como features. Na prática, isso
significa que ações de retenção baseadas nele podem ser aplicadas de forma
diferente entre grupos demográficos. Esse risco precisa ser monitorado antes
de qualquer uso em produção real.

`Contract` é a feature mais preditiva (clientes month-to-month têm 15x mais
churn). O modelo tende a "focar" muito nessa feature, o que pode mascarar
outras causas de cancelamento.

---

## Quando usar e quando não usar

**Use para:**
- Priorizar quais clientes abordar em campanhas de retenção
- Identificar segmentos de risco para análise mais detalhada
- Comparar performance com modelos futuros (esse é o baseline)

**Não use para:**
- Tomar decisões automáticas sem revisão humana
- Penalizar ou restringir serviços de clientes baseado só na predição
- Operadoras com perfil muito diferente do dataset de treino

---

## Monitoramento

Em produção, os sinais mais importantes a acompanhar:

**Performance do modelo:**
- AUC-ROC no conjunto de validação — alertar se cair abaixo de 0.75
- Taxa de churn real vs. previsto mensalmente — desvio acima de 5pp indica
  que o modelo pode estar desatualizado

**Dados de entrada:**
- Distribuição de `MonthlyCharges` e `tenure` — mudanças grandes sugerem
  que o perfil de clientes mudou (concept drift)
- Novas categorias em features categóricas — o `OneHotEncoder` com
  `handle_unknown="ignore"` não vai quebrar, mas vai ignorar silenciosamente
  a nova categoria. Retreino necessário.

**API:**
- Latência do `/predict` — alertar se p95 ultrapassar 500ms
- Taxa de erros 4xx/5xx — acima de 1% das requisições indica problema de
  integração ou dados malformados

**Se o AUC cair abaixo de 0.75:** comparar distribuição dos dados recentes
com os dados de treino, avaliar retreino com dados mais recentes.

---

## Detalhes técnicos

| Item | Valor |
|---|---|
| Framework | PyTorch 2.12 + scikit-learn 1.9 |
| API | FastAPI + Uvicorn |
| Endpoint | `POST /predict` |
| Healthcheck | `GET /health` |
| Arquitetura de deploy | Real-time (por cliente, síncrono) |
| Tempo de inferência | < 50ms (local) |
| Seed | 42 |
| Repositório | github.com/NycolasPantarine/tech-challenge-fase1 |