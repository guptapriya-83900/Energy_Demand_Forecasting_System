import requests
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from scripts.preprocess_data import preprocess_data,save_cleaned_data

#API Configuration
API_KEY= "PcTz1LUYKpW5ci3danfWb6vG86VpTDvu7R3cWMqm"
BASE_URL= "https://api.eia.gov/v2/electricity/rto/region-data/data/"

PARAMS = {
    "frequency": "hourly",                  # Data frequency
    "data[0]": "value",                    # Requested data field
    "sort[0][column]": "period",           # Sort by the period (time)
    "sort[0][direction]": "desc",          # Sort in descending order
    "offset": 0,                           # Starting point for data
    "length": 5000,                        # Number of records to fetch
    "api_key": API_KEY                     # Your EIA API key
}

#Directory Setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Root directory of the script
raw_data=os.path.join(PROJECT_ROOT,"../data/raw")
raw_file_path=os.path.join(raw_data,"energy_data.csv")
processed_file_path=os.path.join(PROJECT_ROOT,"../data/processed")

# Ensure the raw data directory exists
os.makedirs(raw_data, exist_ok=True)

def fetch_data():
    """Fetch hourly energy data from the EIA API and save it as a CSV."""
    print("Fetching data from the EIA API...")

    try:
        response=requests.get(BASE_URL, params=PARAMS)
        response.raise_for_status()  # Raise an error for HTTP codes >= 400
        
        # Parse the JSON response
        data=response.json()
        df=pd.json_normalize(data['response']['data'])

        df.to_csv(raw_file_path,index=False)
        print(f"Data successfully saved to {raw_file_path}")

        # Preprocess the data
        df = preprocess_data(df)

        # Save the cleaned data
        save_cleaned_data(df, processed_file_path)
        print("Preprocessing and saving cleaned data complete.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data: {e}")

if __name__=="__main__":
    fetch_data()

