import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/diabetes.csv"


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Scale features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Set MLflow experiment
    mlflow.set_experiment("Diabetes_Classification")

    with mlflow.start_run():

        # Create and train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters
        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Save model locally
        os.makedirs("models", exist_ok=True)

        mlflow.sklearn.log_model(
            model,
            "model",
        )

        print(f"Model Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
