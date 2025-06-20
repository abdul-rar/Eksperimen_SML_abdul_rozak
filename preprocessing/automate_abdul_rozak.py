import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = '../energy_consumption_raw/train_energy_data.csv'
TEST_PATH = '../energy_consumption_raw/test_energy_data.csv'

def load_and_preprocess_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    """Load and preprocess the energy consumption data."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    combined_df = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

    selected_df = combined_df.drop(columns=['Day of Week', 'Average Temperature'])
    
    # Preprocess Data
    processed_df = selected_df.copy()

    processed_df = pd.concat([processed_df, pd.get_dummies(processed_df['Building Type'], prefix='Building Type')],axis=1)
    processed_df.drop(['Building Type'], axis=1, inplace=True)

    train_df, test_df = train_test_split(processed_df, test_size=0.1, random_state=42)

    numerical_features = ['Square Footage', 'Number of Occupants', 'Appliances Used']
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_features])

    train_df[numerical_features] = scaler.transform(train_df[numerical_features])
    test_df[numerical_features] = scaler.transform(test_df[numerical_features])

    train_df.to_csv('energy_consumption_preprocessing/train_preprocessing.csv', index=False)
    test_df.to_csv('energy_consumption_preprocessing/test_preprocessing.csv', index=False)

    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_and_preprocess_data()
    print("Training and testing data preprocessed and saved successfully.")
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")