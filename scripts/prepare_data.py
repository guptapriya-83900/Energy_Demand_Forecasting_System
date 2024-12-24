import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_column(df, column_name):
    """Normalize a column using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])
    return df, scaler

def create_sequences_for_next_day(data, sequence_length):
    """Create sequences of past data to predict the next day's same hour demand."""
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length - 24):  # Ensure we have data for the next day
        seq = data[i:i + sequence_length]  # Sequence of past `sequence_length` hours
        target = data[i + sequence_length + 24]  # Value for the same hour on the next day
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


#Preparing Data to feed into the network
def prepare_data(file_path, sequence_length=24, test_split=0.2):
    """Prepare the data for training and testing for next-day predictions."""
    # Load the cleaned data
    df = pd.read_csv(file_path)

    # Normalize the 'value' column
    df, scaler = normalize_column(df, 'value')

    # Create sequences
    sequences, targets = create_sequences_for_next_day(df['value'].values, sequence_length)

    # Split into training and test sets
    split_index = int(len(sequences) * (1 - test_split))
    X_train, X_test = sequences[:split_index], sequences[split_index:]
    y_train, y_test = targets[:split_index], targets[split_index:]

    return X_train, X_test, y_train, y_test, scaler
