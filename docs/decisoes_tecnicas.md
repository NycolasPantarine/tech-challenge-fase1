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
- Validação cruzada (quando aplicada) deve ser estratificada pelo mesmo
  motivo.

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
modelo real (Regressão Logística, MLP) precisa superar AUC-ROC = 0.5 de forma
clara para demonstrar que está aprendendo algo útil dos dados.

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
relevante para a análise de trade-off de custo (Etapa 2 do desafio), a ser
revisitado na comparação com a MLP.

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

| Métrica | Dummy | Regressão Logística | MLP |
|---|---|---|---|
| AUC-ROC | 0.50 | 0.8421 | 0.8419 |
| F1 (churn) | 0.00 | 0.60 | 0.60 |
| Recall (churn) | 0.00 | 0.56 | 0.58 |
| Precision (churn) | 0.00 | 0.66 | 0.62 |
| Accuracy | 0.73 | 0.81 | 0.79 |

**Achado principal:** a MLP **não superou** a Regressão Logística de forma
significativa — AUC praticamente idêntico (diferença de 0.0002, dentro da
margem de ruído).

**Análise (trade-off de complexidade vs. ganho):**

1. O problema de churn neste dataset parece ser predominantemente **linear
   ou de baixa interação** — os principais sinais identificados na EDA
   (tenure baixo, contrato mensal) são relações relativamente diretas, que
   um modelo linear já captura bem.

2. Datasets tabulares de tamanho moderado (~7.000 registros, 45 features
   após encoding) frequentemente não se beneficiam da capacidade extra de
   redes neurais — é um padrão conhecido na literatura de ML: para dados
   tabulares, modelos lineares e baseados em árvore costumam ser competitivos
   ou superiores a MLPs.

**Conclusão / recomendação:** para este problema, a Regressão Logística é
preferível como modelo de produção — desempenho equivalente, maior
interpretabilidade, menor custo computacional e maior simplicidade de
manutenção. A MLP permanece no projeto como exigência do desafio e para
demonstrar competência em PyTorch, mas a recomendação técnica documentada
é pelo modelo mais simples.

---

## 10. Trade-off de custo: Falso Positivo vs. Falso Negativo

Com recall ≈ 0.56-0.58 para a classe "churn", entre 42-44% dos clientes que
de fato cancelam **não são identificados** pelo modelo (falsos negativos).

**Custo de negócio:**
- **Falso negativo** (não identificar um cliente que vai cancelar): perda
  direta de receita recorrente — o cliente cancela sem nenhuma ação de
  retenção.
- **Falso positivo** (identificar como risco um cliente que não cancelaria):
  custo de uma ação de retenção (ex: desconto, contato) aplicada

*(documento em construção — próximas seções: Regressão Logística, MLP,
comparação de modelos, etc.)*
