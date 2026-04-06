# OU
code README.md
# OU
notepad.exe README.md

# 👉 Dentro do editor: - Apague TUDO (Ctrl+A, Delete)# 🤖 XGBoost MLOps API — Production-Ready - Cole o README COMPLETO (com a seção de desacoplamento) - Salve (nano: Ctrl+O → Enter 
#    → Ctrl+X)[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)](https://fastapi.tiangolo.com) 
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)](https://mlflow.org) 
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com) 
[![CI/CD](https://github.com/sereno4/mlflow-xgboost-api/actions/workflows/ci.yml/badge.svg)](https://github.com/sereno4/mlflow-xgboost-api/actions) [![License: 
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Render](https://img.shields.io/badge/Deploy-Render-brightgreen?logo=render)](https://xgb-api-agp5.onrender.com)
# 4. Commit + Push
git add -A> Pipeline completo de MLOps: **treino versionado**, **serving com FastAPI**, **monitoramento de drift em tempo real**, **shadow deployment** e **observabilidade com 
Prometheus/Grafana**. git commit -m "docs: README completo com arquitetura desacoplada, shadow deployment e drift monitoring ---

## 📊 Resultados do Modelo
- Adiciona seção de loose coupling com tabela de componentes - Documenta fluxo de predição assíncrono| Métrica | Valor | - Inclui exemplo de ModelLoader com cache e injeção de 
dependência|---------|-------| - Atualiza badges, stack e instruções de deploy"| **Accuracy** | 96% | git push origin main| **Precision** | 97% |
| **ROC-AUC** | 99.2% | **Latência P95** | <50ms | **Drift Detection** | KS Test + PSI |
# 5. Confirmar
git log --oneline -3--- echo "✅ Pronto! Acesse: https://github.com/sereno4/mlflow-xgboost-api/blob/main/README.md"
y## 🏗️ Arquitetura

┌─────────────────────────────────────────┐
│ 🌐 API FastAPI (Porta 8000) │
│ • Predições em tempo real │
│ • Shadow deployment (opcional) │
│ • Métricas Prometheus (/metrics) │
└────────────────┬────────────────────────┘
│
┌────────────▼────────────┐
│ 📈 Observabilidade │
│ • Prometheus: scraping │
│ • Grafana: dashboards │
│ • Alertas de drift │
└────────────┬────────────┘
│
┌────────────▼────────────┐
│ 🧪 MLflow Tracking │
│ • Versionamento de │
│ modelos e parâmetros │
│ • Comparação de runs │
└─────────────────────────┘


---

## 🛠️ Stack Tecnológica

| Categoria | Tecnologias |
|-----------|------------|
| **Linguagem** | Python 3.10 |
| **ML** | XGBoost, scikit-learn, Pandas |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Tracking** | MLflow |
| **Monitoramento** | Prometheus, Grafana |
| **Cache/Filas** | Redis |
| **Infra** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Deploy** | Render (público) |

---

## 🚀 Como Rodar

### 🔹 Opção A: Docker (Recomendado — Stack Completa)
```bash
git clone https://github.com/sereno4/mlflow-xgboost-api.git
cd mlflow-xgboost-api
docker-compose up -d

✅ Serviços disponíveis:
🔹 API: http://localhost:8000/docs
🔹 MLflow: http://localhost:5000
🔹 Prometheus: http://localhost:9092
🔹 Grafana: http://localhost:3002 (admin/admin123)
🔹 Redis: localhost:6381

🌑 Shadow Deployment (Comparação Segura em Produção)
Execute um modelo candidato em paralelo ao principal sem impactar os usuários. Ideal para validar novas versões antes do rollout.
🔹 Como Ativar

# Via variável de ambiente (docker-compose.yml ou .env)
ENABLE_SHADOW=true
SHADOW_MODEL_URI="models:/XGBoost@candidate"

Requisição de Predição
         │
    ┌────▼────┐
    │ Modelo  │ ← Resposta enviada ao cliente (principal)
    │ Principal│
    └────┬────┘
         │
    ┌────▼────┐
    │ Modelo  │ ← Execução em background (shadow)
    │ Shadow  │   • Compara predição
    └────┬────┘   • Calcula disagreement rate
         │
    ┌────▼────┐
    │ Métricas│ ← Exportadas para Prometheus
    │ Shadow  │   • shadow_disagreement_rate
    │         │   • shadow_latency_p95
    └─────────┘

🔹 Métricas Exportadas (Prometheus)

# Exemplo de query no Grafana/Prometheus
shadow_disagreement_rate{model="xgboost"}  # % de previsões divergentes
shadow_prediction_latency_seconds{quantile="0.99"}  # Latência P99 do shadow

Dashboard de Comparação (Grafana)
📊 Side-by-side: Principal vs Shadow
📈 Disagreement rate ao longo do tempo
⚠️ Alerta se divergência > 5%
🔄 Botão "Promover Shadow" (via API admin)
🔹 API Admin para Gerenciamento

# Ativar/desativar shadow mode
curl -X POST http://localhost:8000/admin/shadow/toggle \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Promover modelo shadow para principal
curl -X POST http://localhost:8000/admin/shadow/promote \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Status atual do shadow deployment
curl -s http://localhost:8000/health | jq .shadow
# → {"status":"ok","shadow":true}

📊 Drift Monitoring
Detecta divergência estatística entre dados de treino e produção usando KS Test e Population Stability Index (PSI).
bash

# Gerar relatório de drift
python3 src/analyze_drift.py

# Saída esperada:
# 📈 KS Statistic: 0.12 | PSI: 0.08 | Alert: ✅ OK
# 🚨 Se drift > 30%: alerta para re-treinamento

🔁 Auto-Retrain (Configurável)
O sistema pode disparar re-treinamento automático quando:
drift_ksi_score > 0.30 OU
drift_psi_score > 0.25

Observabilidade
Prometheus Metrics
A API expõe métricas no formato Prometheus em /metrics:

Observabilidade
Prometheus Metrics
A API expõe métricas no formato Prometheus em /metrics:

 API expõe métricas no formato Prometheus em /metrics:
curl http://localhost:8000/metrics
# → prediction_requests_total, prediction_latency_seconds, drift_ksi_score, shadow_disagreement_rate...
Grafana Dashboards
📈 Predições por minuto
⏱️ Latência P95/P99
📉 Drift Score em tempo real
🔀 Shadow vs Principal (quando ativado)
🔄 Auto-retrain events


mlflow-xgboost-api/
├── api/
│   ├── main.py          # FastAPI app + endpoints + shadow logic
│   ├── schemas.py       # Pydantic models
│   └── dependencies.py  # Auth, caching, shadow toggle
├── src/
│   ├── train.py         # Treino + MLflow logging
│   ├── drift_detector.py # KS Test + PSI
│   ├── shadow_runner.py # Execução paralela do modelo shadow
│   └── analyze_drift.py # CLI para relatório
├── monitoring/
│   ├── prometheus.yml   # Config de scraping
│   └── grafana/
│       └── dashboards/  # Dashboards provisionados (inclui shadow panel)
├── tests/
│   ├── test_api.py      # pytest + httpx
│   ├── test_shadow.py   # Validação do shadow deployment
│   └── test_drift.py    # Validação estatística
├── .github/workflows/
│   └── ci.yml           # Build, test, deploy
├── docker-compose.yml   # Stack completa
├── Dockerfile           # Multi-stage build
├── requirements.txt
└── README.md

🔗 Arquitetura Desacoplada

---

## 🔗 Arquitetura Desacoplada (Loose Coupling)

Sistema projetado com **responsabilidades isoladas** para escalabilidade, testabilidade e manutenção.

### 🔹 Princípios de Desacoplamento

| Componente | Responsabilidade | Comunicação |
|------------|-----------------|-------------|
| **API Layer** (`api/main.py`) | Receber requests, validar schemas, retornar respostas | HTTP/JSON |
| **Model Service** (`src/model_loader.py`) | Carregar/gerenciar modelos do MLflow | Interface Python + Cache em memória |
| **Drift Detector** (`src/drift_detector.py`) | Calcular KS Test e PSI de forma assíncrona | Background task + Redis pub/sub |
| **Shadow Runner** (`src/shadow_runner.py`) | Executar modelo candidato em paralelo | Thread pool isolada + métricas Prometheus |
| **Metrics Exporter** (`api/metrics.py`) | Exportar métricas no formato Prometheus | Middleware FastAPI + `/metrics` endpoint |
| **Alerting Service** (`src/alerter.py`) | Disparar alertas/retrain baseado em thresholds | Webhook + logs estruturados |

### 🔹 Fluxo de Predição Desacoplado

### 🔹 Benefícios do Desacoplamento

✅ **Resiliência**: Falha no drift detector não afeta predições  
✅ **Escalabilidade**: Cada componente pode ser escalado independentemente  
✅ **Testabilidade**: Mock de serviços em testes unitários  
✅ **Manutenibilidade**: Trocar modelo/algoritmo sem reescrever a API  
✅ **Observabilidade**: Métricas granulares por componente  

### 🔹 Exemplo: Carregar Modelo sem Acoplar à API

```python
# src/model_loader.py (service isolado)
class ModelLoader:
    def __init__(self, mlflow_uri: str, model_name: str):
        self.client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        self.model_name = model_name
        self._cache = {}  # Cache em memória
    
    def load(self, stage: str = "production") -> XGBClassifier:
        key = f"{self.model_name}@{stage}"
        if key not in self._cache:
            model_uri = f"models:/{self.model_name}@{stage}"
            self._cache[key] = mlflow.sklearn.load_model(model_uri)
        return self._cache[key]

# api/main.py (API usa o service via interface)
model_service = ModelLoader(mlflow_uri=os.getenv("MLFLOW_URI"), model_name="xgboost")

@app.post("/predict")
async def predict(payload: PredictionRequest):
    model = model_service.load()  # ← Injeção de dependência
    return {"prediction": model.predict([payload.features]).tolist()}

 Deploy Público
API disponível em produção:
🔗 https://xgb-api-agp5.onrender.com/docs
⚠️ Render free tier: primeira requisição pode levar ~30s para "acordar".
💡 Shadow mode requer 2x recursos — ative apenas em ambientes com capacidade.




