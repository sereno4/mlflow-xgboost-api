"""Model Loader Service — Carrega modelos do MLflow com cache em memória."""
import logging
import mlflow.sklearn
from typing import Optional, Dict
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, mlflow_uri: str, model_name: str):
        self.mlflow_uri = mlflow_uri
        self.model_name = model_name
        self._cache: Dict[str, XGBClassifier] = {}
        logger.info(f"ModelLoader initialized: {model_name} @ {mlflow_uri}")
    
    def load(self, stage: str = "production") -> XGBClassifier:
        """Carrega modelo do MLflow com cache em memória."""
        cache_key = f"{self.model_name}@{stage}"
        
        if cache_key not in self._cache:
            logger.info(f"Loading model from MLflow: {cache_key}")
            model_uri = f"models:/{self.model_name}@{stage}"
            self._cache[cache_key] = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded and cached: {cache_key}")
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Limpa cache para forçar recarregamento (útil após retrain)."""
        logger.info("Clearing model cache")
        self._cache.clear()
    
    def get_cached_models(self) -> list:
        """Retorna lista de modelos em cache."""
        return list(self._cache.keys())
