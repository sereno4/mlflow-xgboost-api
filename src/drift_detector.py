import numpy as np
import pandas as pd
from scipy import stats
import json, logging, os
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Converte tipos numpy para tipos nativos Python para JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

class DriftDetector:
    def __init__(self, reference_data, feature_names, threshold=0.05, method="ks"):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.threshold = threshold
        self.method = method
        
    def detect_batch_drift(self, current_data):
        results = {
            "features_with_drift": 0, 
            "feature_details": {}, 
            "n_samples": int(len(current_data)), 
            "n_features": int(len(self.feature_names))
        }
        for i, name in enumerate(self.feature_names):
            ref, curr = self.reference_data[:, i], current_data[:, i]
            if self.method == "ks":
                _, p = stats.ks_2samp(ref, curr)
                drift = bool(p < self.threshold)
                results["feature_details"][name] = {"p_value": float(p), "drift_detected": drift}
            else: # PSI simplificado
                hist_ref, bins = np.histogram(ref, bins=10)
                hist_curr, _ = np.histogram(curr, bins=bins)
                hist_ref, hist_curr = hist_ref+1e-5, hist_curr+1e-5
                psi = np.sum((hist_curr/np.sum(hist_curr) - hist_ref/np.sum(hist_ref)) * np.log((hist_curr/np.sum(hist_curr)) / (hist_ref/np.sum(hist_ref))))
                drift = bool(psi >= self.threshold)
                results["feature_details"][name] = {"psi": float(psi), "drift_detected": drift}
            if drift: results["features_with_drift"] += 1
            
        results["drift_rate"] = float(results["features_with_drift"] / results["n_features"])
        results["overall_drift"] = bool(results["drift_rate"] > 0.3)
        results["timestamp"] = pd.Timestamp.now().isoformat()
        return results

    def log_to_mlflow(self, results):
        import mlflow
        # Converter todos os tipos numpy para Python nativo antes de logar
        results_clean = convert_to_serializable(results)
        
        with mlflow.start_run(run_name=f"drift-check-{pd.Timestamp.now().strftime('%Y%m%d')}", nested=True):
            mlflow.log_param("method", self.method)
            mlflow.log_param("threshold", self.threshold)
            mlflow.log_metric("drift_rate", results_clean["drift_rate"])
            mlflow.log_metric("features_with_drift", results_clean["features_with_drift"])
            mlflow.log_param("overall_drift", results_clean["overall_drift"])
            path = "drift_details.json"
            with open(path, "w") as f: 
                json.dump(results_clean["feature_details"], f, indent=2)
            mlflow.log_artifact(path, "drift")
            if os.path.exists(path):
                os.remove(path)
