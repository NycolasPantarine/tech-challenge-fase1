# Decisões Técnicas — Diário de Bordo

Este documento registra as decisões técnicas tomadas durante o desenvolvimento
do projeto, com o porquê de cada uma. Serve como base para o Model Card, o
README e o roteiro do vídeo STAR.

---

## 1. Limpeza de dados — `TotalCharges`

**Problema encontrado:** 11 registros (0,16% do dataset) tinham `TotalCharges`
como string vazia, fazendo a coluna inteira ser lida como `object` em vez de
numérica.

**Investigação:** todos os 11 registros tinham `tenure == 0` (clientes recém
assinados, sem cobrança ainda) e nenhum deles tinha `Churn == "Yes"`.

**Decisão:** preencher esses valores com `0` (em vez de remover as linhas),
porque `TotalCharges = 0` é semanticamente correto para um cliente com
`tenure = 0` — preserva 100% dos registros sem introduzir dados incorretos.

**Resultado:** `TotalCharges` convertido para `float64`, dataset completo
mantido (7.043 registros).

---

## 2. Distribuição da variável target — `Churn`

**Achado:** dataset desbalanceado — 73,5% "No" (não-churn) vs. 26,5% "Yes"
(churn). Proporção aproximada de 3:1.

**Implicações técnicas:**
- Acurácia não é métrica confiável (um modelo que sempre prevê "No" já acerta
  73,5%).
- Métricas escolhidas: AUC-ROC, F1-score, Precision e Recall (sensíveis ao
  desbalanceamento).
- Split treino/teste feito com `stratify=y` para preservar a proporção
  73,5%/26,5% em ambos os conjuntos.
- Validação cruzada estratificada aplicada pelo mesmo motivo.

---

## 3. Insights de EDA relevantes para o negócio

- **`tenure` baixo correlaciona fortemente com churn**: no primeiro mês de
  contrato, a proporção de churn é próxima de 50%. Conforme `tenure` aumenta,
  o churn cai drasticamente. → Sugere que ações de retenção devem ser
  concentradas nos primeiros meses do cliente.

- **Tipo de contrato é fortemente preditivo**:
  - Month-to-month: 42,7% de churn
  - One year: 11,3% de churn
  - Two year: 2,8% de churn
  → Clientes sem fidelidade contratual cancelam ~15x mais que clientes com
  contrato de 2 anos.

---

## 4. Pipeline de pré-processamento

**Decisão:** usar `ColumnTransformer` + `Pipeline` (Scikit-learn) em vez de
transformações manuais (`pd.get_dummies`, normalização manual).

**Motivo:**
- Garante que a mesma transformação aprendida no treino seja aplicada
  consistentemente em dados novos (teste, produção via API).
- Evita problemas de "colunas diferentes" entre treino e inferência.
- `StandardScaler` nas 4 features numéricas (`SeniorCitizen`, `tenure`,
  `MonthlyCharges`, `TotalCharges`) — necessário porque modelos lineares e
  redes neurais são sensíveis à escala.
- `OneHotEncoder(handle_unknown="ignore")` nas 15 features categóricas —
  `handle_unknown="ignore"` evita que a API quebre caso receba uma categoria
  nunca vista no treino.

---

## 5. Baseline 1 — DummyClassifier

**Configuração:** `strategy="most_frequent"` (sempre prevê a classe
majoritária — "não-churn").

**Resultado:**
- Acurácia: 0.73 (enganosa — só reflete a proporção das classes)
- Precision/Recall/F1 da classe "churn" (1): todos 0.0
- **AUC-ROC: 0.5** — equivalente a um classificador aleatório (sem poder
  discriminativo)

**Por que isso importa:** define o piso absoluto de referência. Qualquer
modelo real precisa superar AUC-ROC = 0.5 de forma clara para demonstrar
que está aprendendo algo útil dos dados.

---

## 6. Reprodutibilidade

`random_state=42` fixado em todas as operações com componente aleatório
(`train_test_split`, modelos). Garante que os resultados sejam idênticos em
re-execuções — requisito obrigatório do Tech Challenge.

---

## 7. Baseline 2 — Regressão Logística

**Configuração:** `LogisticRegression(max_iter=1000, random_state=42)`,
usando o mesmo pipeline de pré-processamento do Dummy.

**Resultado:**
- Acurácia: 0.81
- AUC-ROC: **0.84** (vs. 0.50 do Dummy)
- Classe "churn" (1): precision=0.66, recall=0.56, f1=0.60

**Interpretação:** ganho expressivo sobre o Dummy — o modelo captura sinal
real dos dados. O recall de 0.56 significa que 44% dos clientes que
efetivamente cancelam não são identificados (falsos negativos) — ponto
relevante para a análise de trade-off de custo.

---

## 8. Modelo Central — MLP (PyTorch)

**Arquitetura:** `45 → 32 → 16 → 1`, com ReLU entre camadas e Dropout(0.2)
após a primeira camada oculta.

**Configuração de treino:**
- Otimizador: Adam (lr=0.001)
- Loss: BCEWithLogitsLoss (binary cross-entropy)
- Batch size: 32
- Early stopping: patience=10 épocas, monitorando val_loss
- Seeds fixados (`torch.manual_seed(42)`, `np.random.seed(42)`)

**Conjuntos de dados:** split adicional de validação a partir do treino
(treino=4.788 / validação=846 / teste=1.409), todos estratificados.

**Treinamento:** convergiu rapidamente — melhor val_loss (0.4188) atingido
na epoch 6, early stopping disparado na epoch 16.

**Resultado no teste:**
- AUC-ROC: 0.8419
- Accuracy: 0.79
- Classe "churn" (1): precision=0.62, recall=0.58, f1=0.60

---

## 9. Comparação MLP vs. Baselines — Análise Crítica

| Métrica | Dummy | Regressão Logística | Random Forest | Gradient Boosting | MLP |
|---|---|---|---|---|---|
| AUC-ROC | 0.50 | 0.8421 | 0.8185 | 0.8433 | 0.8419 |
| F1 (churn) | 0.00 | 0.60 | 0.55 | 0.58 | 0.60 |
| Recall (churn) | 0.00 | 0.56 | 0.49 | 0.52 | 0.58 |
| Precision (churn) | 0.00 | 0.66 | 0.63 | 0.67 | 0.62 |
| Accuracy | 0.73 | 0.81 | 0.79 | 0.80 | 0.79 |

**Achado principal:** a MLP não superou a Regressão Logística de forma
significativa — AUC praticamente idêntico (diferença de 0.0002).

**Análise:**
1. O problema parece ser predominantemente linear — os principais sinais
   (tenure baixo, contrato mensal) são relações diretas que um modelo linear
   já captura bem.
2. Datasets tabulares de tamanho moderado (~7.000 registros) frequentemente
   não se beneficiam da capacidade extra de redes neurais.

**Conclusão:** para este problema, a Regressão Logística é preferível como
modelo de produção — desempenho equivalente, maior interpretabilidade, menor
custo computacional. A MLP permanece no projeto como exigência do desafio.

---

## 10. Trade-off de custo: Falso Positivo vs. Falso Negativo

Com recall ≈ 0.56-0.58 para a classe "churn", entre 42-44% dos clientes que
de fato cancelam não são identificados pelo modelo (falsos negativos).

**Custo de negócio:**
- **Falso negativo** (não identificar um cliente que vai cancelar): perda
  direta de receita recorrente — o cliente cancela sem nenhuma ação de
  retenção.
- **Falso positivo** (identificar como risco um cliente que não cancelaria):
  custo de uma ação de retenção (ex: desconto, contato) aplicada
  desnecessariamente — custo menor e recuperável.

**Implicação:** dado que o custo de um falso negativo tende a ser maior,
pode ser preferível ajustar o threshold abaixo de 0.5 para aumentar recall
em troca de menor precision.

---

## 11. Ensembles e Validação Cruzada

**Modelos adicionais treinados:** Random Forest (AUC=0.8185) e Gradient
Boosting (AUC=0.8433) — ambos comparados contra MLP e Regressão Logística.

**Resultado:** Gradient Boosting ficou marginalmente acima (AUC=0.8433 vs
0.8421 da Regressão Logística), mas a diferença é irrelevante na prática.
Random Forest ficou abaixo de todos os outros modelos reais.

**Validação cruzada estratificada (5-fold) — Regressão Logística:**
- AUC por fold: [0.8545, 0.8455, 0.8636, 0.8254, 0.8364]
- Média: 0.8451 ± 0.0134

Desvio padrão baixo (0.013) confirma que o modelo generaliza de forma
estável — não depende de um split específico para ter boa performance.

---

## 12. Arquitetura da API — FastAPI + Pydantic

**Decisão:** FastAPI como framework de inferência, com validação de entrada
via Pydantic e schemas tipados.

**Motivo:**
- FastAPI gera documentação interativa automaticamente (`/docs`)
- Pydantic com `Literal` types garante validação explícita de cada campo —
  erros de entrada são rejeitados com 422 antes de chegar ao modelo
- `lifespan` para carregamento dos modelos — artefatos carregados uma única
  vez no startup, não a cada requisição
- Middleware de latência registra tempo de resposta de cada requisição no log

**Decisão de modelo padrão:** `/predict` usa `logreg` por padrão, com
suporte opcional a `model=mlp` via query param.

---

## 13. Estratégia de testes

**3 camadas de teste, 12 testes no total:**

- **Smoke tests** (`test_smoke.py`) — validam que os componentes carregam
  e instanciam corretamente, independente de dados reais.
- **Schema tests** (`test_schema.py`) — validam o contrato dos dados
  processados usando `pandera`.
- **API tests** (`test_api.py`) — testam os endpoints de ponta a ponta,
  cobrindo caminhos felizes e caminhos de erro (400/422).

**Execução:** `make test` roda todos os 12 testes. `make lint` verifica
qualidade com `ruff` (zero erros).

---

## 14. Qualidade de código

**`ruff`** configurado no `pyproject.toml` (regras E, F, I):
- E → erros de estilo (PEP 8), linha máxima de 88 caracteres
- F → erros lógicos (imports não usados, variáveis indefinidas)
- I → ordenação de imports

**`Makefile`** encapsula os comandos principais:
- `make install` → `uv sync`
- `make lint` → `ruff check src/ tests/`
- `make test` → `pytest tests/ -v`
- `make run` → `uvicorn src.api.main:app --reload`

---

## 15. Deploy em produção — GCP Cloud Run

**Plataforma:** Google Cloud Platform — Cloud Run (serverless, managed).

**Motivo da escolha:** Cloud Run é a opção mais simples para FastAPI no GCP —
recebe uma imagem Docker e gera URL pública automaticamente, sem gerenciar
infraestrutura.

**Configuração:**
- Imagem Docker baseada em `python:3.11-slim`
- Memória: 2Gi (necessário para carregar PyTorch)
- Região: us-central1
- Acesso: público (`--allow-unauthenticated`)

**URL pública:** `https://churn-api-311138300643.us-central1.run.app`