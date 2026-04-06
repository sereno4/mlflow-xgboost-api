import os
import logging
import xgboost as xgb
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Schema de request/response (inline, sem importar de schemas.py para evitar dependências)
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=30, max_length=30)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str
    model_version: str = "prod-v1"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Caminho do modelo exportado (já está no repo)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")

# Carregar modelo NA INICIALIZAÇÃO (fora do lifespan, para garantir bind da porta)
logger.info(f"🔍 Carregando modelo de {MODEL_PATH}...")
model = xgb.Booster()
model.load_model(MODEL_PATH)
FEATURE_NAMES = model.feature_names or [f"f{i}" for i in range(30)]
logger.info(f"✅ Modelo carregado | Features: {len(FEATURE_NAMES)}")

# Iniciar app
app = FastAPI(title="XGBoost Prediction API", version="2.0.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", model_loaded=True)

@app.get("/health")
async def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    try:
        # Validar número de features
        if len(req.features) != 30:
            raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(req.features)}")
        
        # Predição
        dmatrix = xgb.DMatrix([req.features], feature_names=FEATURE_NAMES)
        prob = float(model.predict(dmatrix)[0])
        pred = 1 if prob > 0.5 else 0
        
        # Calcular confiança
        conf_score = max(prob, 1 - prob)
        confidence = "alta" if conf_score >= 0.9 else "media" if conf_score >= 0.7 else "baixa"
        
        logger.info(f"🔮 Predição: {pred} | Prob: {prob:.4f} | Conf: {confidence}")
        
        return PredictionResponse(
            prediction=pred,
            probability=round(prob, 4),
            confidence=confidence,
            model_version="prod-v1"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    return {
        "model_type": "xgboost.Booster",
        "feature_count": len(FEATURE_NAMES),
        "loaded": True,
        "version": "prod-v1"
    }

# Entry point para uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
