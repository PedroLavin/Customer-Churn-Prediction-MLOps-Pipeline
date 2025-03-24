import requests
from src.data.ingest_data import preprocess_data, load_data
import pandas as pd

# Test data
df = {
    "CustomerID": 2.0,
    "Age": 30.0,
    "Tenure": 39.0,
    "Usage Frequency": 14.0,
    "Support Calls": 5.0,
    "Payment Delay": 18.0,
    "Total Spend": 932.0,
    "Last Interaction": 17.0,
    "Gender_Female": 1,
    "Gender_Male": 0,
    "Subscription Type_Basic": 0,
    "Subscription Type_Premium": 0,
    "Subscription Type_Standard": 1,
    "Contract Length_Annual": 1,
    "Contract Length_Monthly": 0,
    "Contract Length_Quarterly": 0
}
# Make prediction request
response = requests.post("http://localhost:8000/predict", json=df)
print("Prediction:", response.json())