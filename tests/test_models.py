import pytest
import pandas as pd
from src.data.ingest_data import preprocess_data, load_data, preprocess_data
from src.models.train import train_model

def test_preprocessing():
    # Create sample data
    sample_data = pd.DataFrame({
    'CustomerID': [2], 
    'Age': [30], 
    'Gender': ['Female'], 
    'Tenure': [39], 
    'Usage Frequency': [14], 
    'Support Calls': [5], 
    'Payment Delay': [18], 
    'Subscription Type': ['Standard'], 
    'Contract Length': ['Annual'], 
    'Total Spend': [932], 
    'Last Interaction': [17]
})
    
    processed_data = preprocess_data(sample_data)
    assert not processed_data.isnull().any().any()
    assert 'gender_Female' in processed_data.columns

def test_model_training():
    try:
        train_model()
        assert True
    except Exception as e:
        assert False, f"Model training failed with error: {e}"