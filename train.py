import os
import joblib
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data.generate_data import generate_dataset


features = [
    "vibration_mm_s",
    "temperature_C",
    "rpm",
    "load_pct",
    "hours_since_maintenance",
]
model_dir = "models"
model_path = os.path.join(model_dir, "model.joblib")

mlflow.set_experiment("predictive-maintenance")

def main():
    os.makedirs(model_dir, exist_ok=True)

    df = generate_dataset(n_samples=10000, seed=42)
    X = df[features]
    y = df["failure"]

    # Split dataset into training and testing sets, split 10,000 rows into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),  # handle class imbalance
        "eval_metric": "logloss",
        "random_state": 42,
    }

    # opens a new mlflow run
    with mlflow.start_run() as run:
        # saves model's settings (max depth, learning rate, etc.) to mlflow
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }

        # saves model's metrics (accuracy, precision, recall, f1, roc_auc)
        mlflow.log_metrics(metrics)
        # saves a full copy of trained model to mlflow's own storage
        mlflow.xgboost.log_model(model, artifact_path="model")

        # saves a second copy of trained model to local storage, along with the features used for training
        joblib.dump({"model": model, "features": features}, model_path)

        print(f"Run ID: {run.info.run_id}")
        print("Model metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
