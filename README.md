# 🚀 XGBoost API com MLflow e Drift Monitoring
[![CI](https://github.com/sereno4/mlflow-xgboost-api/actions/workflows/ci.yml/badge.svg)](https://github.com/sereno4/mlflow-xgboost-api/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Pipeline de MLOps: treino, tracking, serving e monitoramento de drift.

## 📋 Resultados
| Accuracy | Precision | ROC-AUC | Latência |
|----------|-----------|---------|----------|
| 96% | 97% | 99.2% | <50ms |

## 🛠️ Stack
Python 3.10 | XGBoost | MLflow | FastAPI | Docker | GitHub Actions

## 🚀 Como Rodar
```bash
git clone https://github.com/sereno4/mlflow-xgboost-api.git && cd mlflow-xgboost-api
pip install -r requirements.txt
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db &
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Acesse: 🔹 API: http://localhost:8000/docs | 🔹 MLflow: http://localhost:5000

## 📊 Drift Monitoring
Detecta divergência entre dados de treino e produção (KS Test/PSI).
```bash
python3 src/analyze_drift.py  # 📈 RELATÓRIO DRIFT | Alerta: 🚨 SIM
```

## 🗂️ Estrutura
```
api/          # FastAPI + Pydantic
src/          # Treino + DriftDetector
tests/        # pytest
.github/      # CI/CD
Dockerfile    # Deploy
```

## 🔐 Segurança
Nunca commite .env ou credenciais. Use PAT com escopo repo.

---
**[@sereno4](https://github.com/sereno4)** | MIT License
