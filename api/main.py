from typing import List, Optional
import os
import logging
import mlflow.xgboost
import xgboost as xgb
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .schemas import PredictionRequest, PredictionResponse, HealthResponse

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variáveis de ambiente
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "3a0452332cc74f2d978ed336782764ef")

# Modelo global
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carregar modelo na inicialização"""
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.xgboost.load_model(f"runs:/{MODEL_RUN_ID}/modelo_xgboost")
        logger.info(f"✅ Modelo carregado com sucesso da run: {MODEL_RUN_ID[:8]}...")
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        raise
    yield
    # Cleanup (se necessário)
    logger.info("👋 Shutting down API...")

# Inicializar FastAPI
app = FastAPI(
    title="XGBoost Prediction API",
    description="API de predição com modelo XGBoost treinado e versionado via MLflow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS (permitir requisições de outros domínios)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique os domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check da API"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI
    )

@app.get("/health")
async def health():
    """Endpoint simples de saúde"""
    return {"status": "ok", "model": "loaded" if model else "not_loaded"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Faz predição usando o modelo XGBoost carregado.
    
    - **features**: Lista de 30 features numéricas
    - **return**: Predição (0 ou 1), probabilidade e confiança
    """
    try:
        # Converter para numpy array
        features_array = np.array(request.features).reshape(1, -1)
        
        # Criar DMatrix para XGBoost
        dmatrix = xgb.DMatrix(features_array)
        
        # Fazer predição
        prob = float(model.predict(dmatrix)[0])
        prediction = 1 if prob > 0.5 else 0
        
        # Calcular confiança
        confidence_score = max(prob, 1 - prob)
        if confidence_score >= 0.9:
            confidence = "alta"
        elif confidence_score >= 0.7:
            confidence = "media"
        else:
            confidence = "baixa"
        
        logger.info(f"🔮 Predição: {prediction} | Prob: {prob:.4f} | Conf: {confidence}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(prob, 4),
            confidence=confidence,
            model_version=MODEL_RUN_ID
        )
        
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predição: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Predição em batch (múltiplas amostras)"""
    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return {"predictions": results, "count": len(results)}

@app.get("/model/info")
async def model_info():
    """Informações do modelo carregado"""
    return {
        "run_id": MODEL_RUN_ID,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "type": "xgboost.Booster",
        "loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
