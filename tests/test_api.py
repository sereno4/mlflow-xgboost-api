import pytest
from fastapi.testclient import TestClient
import sys
import os

# Adicionar a raiz do projeto ao path para importar a api
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock do MLflow para não precisar do banco na CI
# (Isso impede que a CI falhe se não conseguir conectar no localhost:5000)
import unittest.mock as mock

@pytest.fixture
def client():
    # Patch para simular o carregamento do modelo sem precisar do MLflow real
    with mock.patch('api.main.mlflow.xgboost.load_model') as mock_load:
        # Retorna um mock object que responde predict()
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = [0.9]
        mock_load.return_value = mock_model
        
        # Simular feature names
        with mock.patch('api.main.FEATURE_NAMES', new=["feat1"] * 30):
            from api.main import app
            with TestClient(app) as c:
                yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_success(client):
    # 30 features (dummy)
    payload = {"features": [0.5] * 30}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_length(client):
    payload = {"features": [0.5] * 10} # Errado (tem que ser 30)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Erro de validação
