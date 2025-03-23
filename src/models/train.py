import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import joblib

def load_training_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    return X_train, y_train

def train_model():
    mlflow.set_experiment("churn_prediction")
    
    X_train, y_train = load_training_data()
    
    with mlflow.start_run():
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train.values.ravel())
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        
        # Save model
        joblib.dump(rf_model, 'models/random_forest_model.joblib')
        mlflow.sklearn.log_model(rf_model, "model")
        
        # Log metrics
        train_score = rf_model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)

if __name__ == "__main__":
    train_model()