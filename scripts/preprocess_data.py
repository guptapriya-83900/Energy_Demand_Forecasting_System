import pandas as pd
import os

def preprocess_data(df):
    """Preprocess the raw energy data."""
    print("Preprocessing data...")
    
    # Convert 'period' to datetime
    df['period'] = pd.to_datetime(df['period'])
    # Remove rows with negative 'value'
       # Convert 'value' column to numeric (forcing errors to NaN)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Remove rows with missing or invalid 'value'
    df = df.dropna(subset=['value'])
    
    # Remove rows with negative 'value'
    df = df[df['value'] >= 0]
    # Handle outliers: Remove rows where 'value' is too high
    threshold = 120000  # Define a threshold for outliers
    df = df[df['value'] <= threshold]
    
    # Extract time features
    df['hour'] = df['period'].dt.hour
    df['day'] = df['period'].dt.day
    df['weekday'] = df['period'].dt.weekday
    
    # Remove unnecessary columns
    df = df.drop(columns=['respondent', 'respondent-name', 'value-units', 'type', 'type-name'])
    
    print("Preprocessing complete.")
    return df

def save_cleaned_data(df, raw_dir):
    """Save cleaned data to a CSV."""
    cleaned_file_path = os.path.join(raw_dir, "cleaned_energy_data.csv")
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data successfully saved to {cleaned_file_path}")
