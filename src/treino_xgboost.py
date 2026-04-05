import os
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURAÇÕES =================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("XGBoost-Classification")

# Hiperparâmetros do XGBoost (logue tudo para comparar depois)
HPARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "n_jobs": -1,  # Usa todos os cores da CPU
    "early_stopping_rounds": 10,
    "eval_metric": "logloss"
}

# ================= PREPARAÇÃO DOS DADOS =================
def prepare_data(use_custom_csv=False, csv_path=None, target_col=None, feature_cols=None):
    """
    Prepara dados para treino.
    
    Args:
        use_custom_csv: Se True, carrega do CSV especificado
        csv_path: Caminho para o CSV (se use_custom_csv=True)
        target_col: Nome da coluna target (se use_custom_csv=True)
        feature_cols: Lista de colunas para features (se use_custom_csv=True)
    """
    if use_custom_csv and csv_path:
        # Carregar dados personalizados
        df = pd.read_csv(csv_path)
        print(f"📊 Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Separar features e target
        if feature_cols:
            X = df[feature_cols]
        else:
            X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Codificar target se for categórico string
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            joblib.dump(le, "label_encoder.pkl")
            mlflow.log_artifact("label_encoder.pkl", "preprocessing")
            if os.path.exists("label_encoder.pkl"):
                os.remove("label_encoder.pkl")
    else:
        # Dados de exemplo (Iris para classificação multiclasse)
        data = load_breast_cancer()  # Mais desafiador que Iris
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        print(f"📊 Usando dataset de exemplo: {X.shape[0]} linhas, {X.shape[1]} features")
    
    # Split estratificado (mantém proporção das classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=HPARAMS["random_state"], stratify=y
    )
    
    print(f"✅ Dados preparados: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()

# ================= TREINAMENTO =================
def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """Treina modelo XGBoost com early stopping e logging"""
    
    # Converter para DMatrix (formato otimizado do XGBoost)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    
    # Parâmetros do modelo
    params = {
        "objective": "binary:logistic" if len(np.unique(y_train)) == 2 else "multi:softprob",
        "num_class": len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else None,
        **{k: v for k, v in HPARAMS.items() if k not in ["early_stopping_rounds", "eval_metric", "n_jobs", "random_state"]}
    }
    # Remover None do params
    params = {k: v for k, v in params.items() if v is not None}
    
    print("🌲 Treinando XGBoost...")
    
    # Treinar com early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=HPARAMS["n_estimators"],
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=HPARAMS["early_stopping_rounds"],
        verbose_eval=10
    )
    
    return model

# ================= AVALIAÇÃO =================
def evaluate_model(model, X_test, y_test, feature_names, is_binary=True):
    """Avalia o modelo e retorna métricas completas"""
    
    # Predições
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    y_pred_proba = model.predict(dtest)
    
    if is_binary:
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Métricas principais
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    
    # ROC-AUC (apenas para binário ou multiclasse com probas)
    if is_binary:
        metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
    else:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
        except:
            pass
    
    # Relatório de classificação (texto)
    report = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, report, cm, y_pred, y_pred_proba

# ================= VISUALIZAÇÃO =================
def plot_feature_importance(model, feature_names, save_path="feature_importance.png"):
    """Gera e salva gráfico de importância de features"""
    plt.figure(figsize=(10, 6))
    
    # Extrair importância
    importance = model.get_score(importance_type="gain")
    features = list(importance.keys())
    scores = list(importance.values())
    
    # Ordenar e pegar top 15
    top_n = min(15, len(features))
    idx = np.argsort(scores)[-top_n:]
    
    plt.barh(range(top_n), [scores[i] for i in idx])
    plt.yticks(range(top_n), [features[i] for i in idx])
    plt.xlabel("Gain (Importância)")
    plt.title("Top Features - XGBoost")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path

def plot_confusion_matrix(cm, class_names=None, save_path="confusion_matrix.png"):
    """Gera e salva matriz de confusão"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path

# ================= MAIN =================
def main(use_custom_csv=False, csv_path=None, target_col=None, feature_cols=None):
    print("🚀 Iniciando treino XGBoost com MLflow tracking...")
    
    # Preparar dados
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        use_custom_csv=use_custom_csv,
        csv_path=csv_path,
        target_col=target_col,
        feature_cols=feature_cols
    )
    
    is_binary = len(np.unique(y_train)) == 2
    class_names = [f"Classe_{i}" for i in np.unique(y_train)]
    
    with mlflow.start_run(run_name=f"xgb-{HPARAMS['max_depth']}d-{HPARAMS['n_estimators']}est"):
        # 1. Logar hiperparâmetros
        mlflow.log_params(HPARAMS)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("n_classes", len(np.unique(y_train)))
        mlflow.log_param("is_binary", is_binary)
        
        # 2. Treinar modelo
        model = train_xgboost(X_train, X_test, y_train, y_test, feature_names)
        
        # 3. Avaliar
        print("🔍 Avaliando modelo...")
        metrics, report, cm, y_pred, y_pred_proba = evaluate_model(
            model, X_test, y_test, feature_names, is_binary
        )
        
        # 4. Logar métricas principais
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        print(f"📈 Métricas: {', '.join([f'{k}={v:.3f}' for k,v in metrics.items()])}")
        
        # 5. Logar modelo XGBoost (formato nativo)
        mlflow.xgboost.log_model(model, "modelo_xgboost")
        
        # 6. Salvar e logar artefatos visuais
        print("📊 Gerando visualizações...")
        
        # Feature importance
        feat_plot = plot_feature_importance(model, feature_names)
        mlflow.log_artifact(feat_plot, "plots")
        if os.path.exists(feat_plot):
            os.remove(feat_plot)
        
        # Confusion matrix
        cm_plot = plot_confusion_matrix(cm, class_names)
        mlflow.log_artifact(cm_plot, "plots")
        if os.path.exists(cm_plot):
            os.remove(cm_plot)
        
        # Relatório de classificação como texto
        with open("classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt", "reports")
        if os.path.exists("classification_report.txt"):
            os.remove("classification_report.txt")
        
        # 7. Salvar feature names para inferência futura
        with open("feature_names.json", "w") as f:
            import json
            json.dump(feature_names, f)
        mlflow.log_artifact("feature_names.json", "metadata")
        if os.path.exists("feature_names.json"):
            os.remove("feature_names.json")
        
        # 8. Cross-validation score (opcional, mas útil)
        try:
            from sklearn.model_selection import cross_val_score
            # Converter para sklearn API compatível
            sklearn_model = model
            cv_scores = cross_val_score(
                sklearn_model, X_train, y_train, cv=5, 
                scoring="accuracy" if is_binary else "f1_weighted"
            )
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())
            print(f"🔄 CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        except Exception as e:
            print(f"⚠️ Cross-validation pulado: {e}")
        
        print(f"\n✅ Run concluída!")
        print(f"🏆 Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f}")
        if "roc_auc" in metrics:
            print(f"📊 ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"🔗 Veja detalhes em: http://localhost:5000")
        print(f"💡 Para carregar o modelo depois:")
        print(f"   model = mlflow.xgboost.load_model(f'runs:/{{run_id}}/modelo_xgboost')")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Treinar XGBoost com MLflow")
    parser.add_argument("--csv", type=str, help="Caminho para CSV personalizado")
    parser.add_argument("--target", type=str, help="Nome da coluna target")
    parser.add_argument("--features", type=str, nargs="+", help="Colunas para features")
    
    args = parser.parse_args()
    
    main(
        use_custom_csv=args.csv is not None,
        csv_path=args.csv,
        target_col=args.target,
        feature_cols=args.features
    )
