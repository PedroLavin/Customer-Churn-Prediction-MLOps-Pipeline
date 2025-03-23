from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load('models/random_forest_model.joblib')

@app.post("/predict")
async def predict(customer_data: dict):
    df = pd.DataFrame([customer_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "churn_prediction": bool(prediction),
        "churn_probability": float(probability)
    }