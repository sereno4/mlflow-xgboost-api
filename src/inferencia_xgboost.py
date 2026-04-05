import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
import json

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def load_best_model(experiment_name="XGBoost-Classification", metric="accuracy"):
    """Carrega o melhor modelo baseado em uma métrica"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if runs.empty:
        raise ValueError("Nenhuma run encontrada")
    
    run_id = runs.iloc[0]["run_id"]
    print(f"📦 Carregando modelo da run: {run_id} ({metric}: {runs.iloc[0][f'metrics.{metric}']:.3f})")
    
    # Carregar modelo
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/modelo_xgboost")
    
    # Carregar feature names (se salvou)
    try:
        artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/metadata/feature_names.json"
        )
        with open(artifact_path, "r") as f:
            feature_names = json.load(f)
    except:
        feature_names = None
    
    return model, feature_names, run_id

def predict(model, X, feature_names=None, is_binary=True):
    """Faz predições com o modelo"""
    # Converter para DMatrix se for XGBoost nativo
    if isinstance(model, xgb.Booster):
        dX = xgb.DMatrix(X, feature_names=feature_names)
        pred_proba = model.predict(dX)
        if is_binary:
            return (pred_proba > 0.5).astype(int), pred_proba
        else:
            return np.argmax(pred_proba, axis=1), pred_proba
    else:
        # Se for sklearn API
        pred = model.predict(X)
        pred_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        return pred, pred_proba

# ================= DEMO =================
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    
    print("🔮 Carregando melhor modelo...")
    model, feature_names, run_id = load_best_model()
    
    print("\n📊 Fazendo predições em 5 amostras de teste...")
    data = load_breast_cancer()
    X_demo = pd.DataFrame(data.data[:5], columns=data.feature_names)
    
    preds, probas = predict(model, X_demo, feature_names, is_binary=True)
    
    print(f"\n{'Amostra':<8} {'Pred':<6} {'Prob':<10} {'Classe Real'}")
    print("-" * 40)
    for i, (p, prob, real) in enumerate(zip(preds, probas, data.target[:5])):
        print(f"{i+1:<8} {p:<6} {prob:.2%}    {data.target_names[real]}")
    
    print(f"\n✅ Inferência concluída! Modelo da run: {run_id[:8]}...")
