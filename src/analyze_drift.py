#!/usr/bin/env python3
import os, sys, json, glob, numpy as np, pandas as pd
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.drift_detector import DriftDetector
from src.treino_xgboost import prepare_data
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def load_production_data(hours=24):
    monitoring_path = "data/monitoring/"
    files = glob.glob(f"{monitoring_path}predictions_*.jsonl")
    if not files: return None
    records = []
    cutoff = datetime.now() - timedelta(hours=hours)
    for file in files:
        with open(file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                if datetime.fromisoformat(entry["timestamp"]) >= cutoff and entry["status_code"] == 200:
                    records.append(entry["features"])
    return np.array(records) if len(records) >= 10 else None

def main():
    print("🔍 Analisando drift...")
    X_train, _, _, _, feature_names = prepare_data()
    prod_data = load_production_data(hours=24)
    if prod_data is None:
        print("❌ Dados insuficientes para análise (<10 amostras).")
        return 0
        
    detector = DriftDetector(X_train.values, feature_names, threshold=0.05, method="ks")
    results = detector.detect_batch_drift(prod_data)
    detector.log_to_mlflow(results)
    
    print(f"\n📈 RELATÓRIO DRIFT | Taxa: {results['drift_rate']:.2%} | Alerta: {'🚨 SIM' if results['overall_drift'] else '✅ NÃO'}")
    top_drift = sorted(results["feature_details"].items(), key=lambda x: x[1]["p_value"])[:5]
    for feat, data in top_drift:
        print(f"  • {feat}: p={data['p_value']:.4f} {'⚠️' if data['drift_detected'] else '✓'}")
    return 1 if results["overall_drift"] else 0

if __name__ == "__main__":
    sys.exit(main())
