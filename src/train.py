import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Conectar no seu servidor local
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MeuProjeto-Teste")

# 2. Dados simples (Iris)
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# 3. Treinar
with mlflow.start_run():
    params = {"n_estimators": 10, "max_depth": 3}
    mlflow.log_params(params)  # Salva hiperparâmetros

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    mlflow.log_metric("acuracia", acc)  # Salva métricas

    mlflow.sklearn.log_model(model, "modelo")  # Salva o arquivo do modelo

    print(f"✅ Sucesso! Acurácia: {acc:.2f}")
    print(f"👀 Veja o resultado em http://localhost:5000")
