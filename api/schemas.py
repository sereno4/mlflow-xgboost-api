from pydantic import BaseModel, Field
from typing import List, Optional
import json

class PredictionRequest(BaseModel):
    """Schema para requisição de predição"""
    features: List[float] = Field(
        ..., 
        description="Lista de 30 features numéricas (dataset breast cancer)",
        min_items=30,
        max_items=30,
        example=[1.2, 0.5, 2.1, 0.8, 1.5, 0.3, 2.0, 1.1, 0.9, 1.7,
                 0.6, 1.3, 0.4, 1.8, 0.7, 1.4, 0.2, 1.9, 1.0, 0.8,
                 1.6, 0.5, 1.2, 0.9, 1.5, 0.4, 1.7, 0.6, 1.3, 0.8]
    )

class PredictionResponse(BaseModel):
    """Schema para resposta de predição"""
    prediction: int = Field(..., description="Classe predita (0=benigno, 1=maligno)")
    probability: float = Field(..., description="Probabilidade da classe positiva")
    confidence: str = Field(..., description="Nível de confiança (baixa/média/alta)")
    model_version: str = Field(..., description="ID da run do MLflow")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.94,
                "confidence": "alta",
                "model_version": "3a0452332cc74f2d978ed336782764ef"
            }
        }

class HealthResponse(BaseModel):
    """Schema para health check"""
    status: str
    model_loaded: bool
    mlflow_tracking_uri: str
    version: str = "1.0.0"
