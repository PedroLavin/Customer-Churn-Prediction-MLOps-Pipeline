import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    # For demo, you can use a sample dataset like this:
    df = pd.read_csv('data/raw/customer_data.csv')
    return df

def preprocess_data(df):
    # Basic preprocessing
    df = df.dropna()
    
    # Feature engineering
    df['Tenure'] = pd.to_numeric(df['Tenure'])
    # df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    
    # Convert categorical variables
    categorical_columns = ['Gender', 'Subscription Type', 'Contract Length']
    df = pd.get_dummies(df, columns=categorical_columns)
    print(df.columns.tolist())
    
    return df

def split_and_save_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed datasets
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

if __name__ == "__main__":
    df = load_data()
    df_processed = preprocess_data(df)
    split_and_save_data(df_processed)