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

*(documento em construção — próximas seções: Regressão Logística, MLP,
comparação de modelos, etc.)*
