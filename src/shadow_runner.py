"""Shadow Deployment Runner — Executa modelo candidato em paralelo."""
import logging
from typing import Dict, List, Optional
import mlflow.sklearn
from src.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class ShadowRunner:
    def __init__(self, mlflow_uri: str, shadow_model_name: str, shadow_stage: str = "candidate"):
        self.loader = ModelLoader(mlflow_uri, shadow_model_name)
        self.shadow_stage = shadow_stage
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            self._model = self.loader.load(stage=self.shadow_stage)
        return self._model
    
    def predict(self, features: List[float]) -> Dict:
        """Executa predição shadow (não bloqueante, sem afetar resposta principal)."""
        try:
            model = self._load_model()
            prediction = model.predict([features])[0]
            proba = model.predict_proba([features])[0].tolist() if hasattr(model, "predict_proba") else None
            return {"prediction": int(prediction), "proba": proba, "status": "ok"}
        except Exception as e:
            logger.error(f"Shadow prediction failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def compare(self, main_pred: int, shadow_result: Dict) -> Dict:
        """Compara predição principal vs shadow e retorna métricas."""
        if shadow_result.get("status") != "ok":
            return {"disagreement": None, "error": shadow_result.get("error")}
        
        disagreement = (main_pred != shadow_result["prediction"])
        return {
            "disagreement": disagreement,
            "main_prediction": main_pred,
            "shadow_prediction": shadow_result["prediction"],
            "shadow_proba": shadow_result.get("proba")
        }
