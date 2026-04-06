#!/usr/bin/env python3
"""
Verifica drift no MLflow e retreina automaticamente se drift_rate > threshold.
"""
import sys
import os
import subprocess
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import update_active_run_id

# Configurações
DRIFT_THRESHOLD = 0.30  # 30%
TRAINING_EXP_NAME = "XGBoost-Classification"
DRIFT_EXP_NAME = "drift_monitoring"
MLFLOW_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

def get_latest_metric(exp_name: str, metric_name: str) -> float:
    """Busca a métrica mais recente de um experimento"""
    exp = client.get_experiment_by_name(exp_name)
    if not exp:
        print(f"⚠️ Experimento '{exp_name}' não encontrado")
        return -1.0
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"metrics.{metric_name} >= 0",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        print(f"⚠️ Nenhuma run encontrada em '{exp_name}'")
        return -1.0
    return runs[0].data.metrics.get(metric_name, -1.0)

def trigger_training() -> str:
    """Executa o script de treino e retorna o novo run_id"""
    print("🔄 Iniciando retreinamento...")
    result = subprocess.run(
        [sys.executable, "src/treino_xgboost.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"❌ Falha no treino:\n{result.stderr}")
    
    # Busca a run mais recente do experimento de treino
    train_exp = client.get_experiment_by_name(TRAINING_EXP_NAME)
    latest = client.search_runs(
        experiment_ids=[train_exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )[0]
    return latest.info.run_id

def main():
    print(f"🔍 Verificando drift (threshold: {DRIFT_THRESHOLD:.0%})...")
    drift_rate = get_latest_metric(DRIFT_EXP_NAME, "drift_rate")
    
    if drift_rate < 0:
        print("⚠️ Sem dados de drift suficientes. Encerrando.")
        return
    
    print(f"📊 Último drift_rate: {drift_rate:.2%}")
    
    if drift_rate > DRIFT_THRESHOLD:
        print(f"🚨 DRIFT ACIMA DO THRESHOLD! Disparando retreinamento...")
        try:
            new_run_id = trigger_training()
            update_active_run_id(new_run_id)
            
            # Loga evento de retreinamento no MLflow
            with mlflow.start_run(run_name=f"auto-retrain-{datetime.now().strftime('%Y%m%d_%H%M')}", 
                                  experiment_id=client.get_experiment_by_name(DRIFT_EXP_NAME).experiment_id):
                mlflow.log_param("trigger", "drift_threshold")
                mlflow.log_metric("drift_rate_at_trigger", drift_rate)
                mlflow.log_param("new_model_run_id", new_run_id)
                mlflow.log_param("status", "success")
            
            print(f"✅ RETREINAMENTO CONCLUÍDO | Novo modelo: {new_run_id}")
            print("💡 Reinicie a API ou aguarde o próximo deploy para aplicar o novo modelo.")
            
        except Exception as e:
            print(f"❌ Erro no retreinamento: {e}")
            # Loga falha
            with mlflow.start_run(run_name=f"auto-retrain-fail-{datetime.now().strftime('%Y%m%d_%H%M')}",
                                  experiment_id=client.get_experiment_by_name(DRIFT_EXP_NAME).experiment_id):
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
    else:
        print("✅ Drift dentro do limite. Nenhum retreinamento necessário.")

if __name__ == "__main__":
    main()
