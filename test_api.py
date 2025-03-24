import requests
from src.data.ingest_data import preprocess_data
import pandas as pd

# Test data
test_data = {
    'CustomerID': 2, 
    'Age': 29, 
    'Tenure': 39, 
    'Usage Frequency': 14, 
    'Support Calls': 5, 
    'Payment Delay': 18, 
    'Total Spend': 932, 
    'Last Interaction': 17,
    'Gender_Female': True,
    'Subscription Type_Premium': True, 
    'Contract Length_Quarterly': True,
    'Gender_Male': False,
    'Subscription Type_Basic': False, 
    'Subscription Type_Standard': False, 
    'Contract Length_Annual': False, 
    'Contract Length_Monthly': False
}

test_d = pd.DataFrame([{
    'CustomerID': 2, 
    'Age': 29, 
    'Gender': 'Female',
    'Tenure': 39, 
    'Usage Frequency': 14, 
    'Support Calls': 5, 
    'Payment Delay': 18, 
    'Subscription Type': 'Premium', 
    'Contract Length': "Quarterly",
    'Total Spend': 932, 
    'Last Interaction': 17
}])
test_df = preprocess_data(test_d)
print(test_df)
# Make prediction request
response = requests.post("http://localhost:8000/predict", json=test_data)
print("Prediction:", response.json())