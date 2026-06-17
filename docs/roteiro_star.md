S — Situation (45 segundos)
"Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. A diretoria precisa identificar quais clientes têm risco de cancelamento para agir antes que isso aconteça. O dataset disponível tem 7.043 clientes com 19 features — dados de contrato, serviços contratados e informações demográficas. O desafio técnico começa já na análise dos dados: 73,5% dos clientes não cancelaram e apenas 26,5% cancelaram — um dataset desbalanceado, onde acurácia vira uma métrica inútil."

T — Task (30 segundos)
"A tarefa foi construir um pipeline completo de ML engineering: da exploração dos dados até um modelo servido via API em produção, aplicando boas práticas de engenharia — código modular, testes automatizados, tracking de experimentos e deploy em nuvem."

A — Action (2 minutos e 30 segundos)
"Primeiro, a EDA revelou dois insights principais: clientes no primeiro mês de contrato têm quase 50% de chance de cancelar, e clientes com contrato mensal cancelam 15 vezes mais que clientes com contrato de 2 anos. Isso orientou tanto as features usadas quanto as métricas escolhidas — AUC-ROC e F1, não acurácia.
Treinamos quatro modelos: DummyClassifier como piso de referência (AUC 0.50), Regressão Logística (AUC 0.84), Random Forest e Gradient Boosting, e a MLP em PyTorch — arquitetura 45→32→16→1, com ReLU, Dropout e early stopping. Todos os experimentos foram rastreados no MLflow.
O resultado mais interessante: a MLP chegou ao mesmo AUC da Regressão Logística (0.8419 vs 0.8421 — diferença irrelevante). Para dados tabulares de 7 mil registros, a complexidade extra da rede neural não trouxe ganho mensurável. A recomendação técnica é pelo modelo mais simples.
Todo o código foi refatorado de notebooks para módulos em src/, com uma API FastAPI com validação Pydantic, logging estruturado, middleware de latência e 12 testes automatizados (smoke, schema e API). Ruff sem erros, Makefile para automação. Deploy no GCP Cloud Run com URL pública."

R — Result (1 minuto e 15 segundos)
"O projeto entrega um pipeline end-to-end funcional: dados brutos entram, predição de churn sai via API. AUC-ROC de 0.84 — partindo de 0.50 do baseline aleatório. Validação cruzada de 5 folds confirma estabilidade: 0.8451 ± 0.013.
A principal lição aprendida é técnica: nem sempre o modelo mais complexo é o melhor. Para esse problema tabular, a Regressão Logística compete de igual para igual com uma rede neural — e em produção, simplicidade tem valor real: menor custo, mais fácil de manter, mais fácil de explicar.
O projeto está disponível no GitHub e a API está no ar no GCP Cloud Run."