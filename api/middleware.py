import time
import json
import numpy as np
from fastapi import Request, Response
from typing import Callable
import logging

logger = logging.getLogger(__name__)

class DriftMonitoringMiddleware:
    """Middleware para coletar features e logar para monitoramento"""
    
    def __init__(self, app, storage_path: str = "data/monitoring/"):
        self.app = app
        self.storage_path = storage_path
        import os
        os.makedirs(storage_path, exist_ok=True)
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # Processar request normalmente
        response = await call_next(request)
        
        # Apenas monitorar endpoint de predição
        if request.url.path == "/predict" and request.method == "POST":
            try:
                # Ler body da request
                body = await request.json()
                features = body.get("features", [])
                
                # Logar para arquivo (batch para eficiência)
                self._log_prediction(features, response.status_code)
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao logar para monitoring: {e}")
        
        return response
    
    def _log_prediction(self, features: list, status_code: int):
        """Salvar features em arquivo para análise posterior de drift"""
        import pandas as pd
        from datetime import datetime
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "status_code": status_code
        }
        
        # Append em arquivo JSONL (JSON Lines)
        log_file = f"{self.storage_path}predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
