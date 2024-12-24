import os
import sys
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from models.cnn_lstm_model import CNNLSTM
from scripts.prepare_data import prepare_data

# Path to cleaned data
cleaned_data_path = "data/processed/cleaned_energy_data.csv"
sequence_length = 24

# Load data
X_train, X_test, y_train, y_test, scaler = prepare_data(cleaned_data_path, sequence_length)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    cnn_filters = trial.suggest_int("cnn_filters", 8, 64, step=8)
    lstm_hidden_size = trial.suggest_int("lstm_hidden_size", 16, 128, step=16)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    # Initialize model
    model = CNNLSTM(
        input_size=1,
        cnn_filters=cnn_filters,
        lstm_hidden_size=lstm_hidden_size,
        num_layers=2,  # Keep number of LSTM layers fixed for now
        output_size=1,
        dropout=dropout,
    )
    model.train()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10  # Keep it short for tuning
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    # Return validation loss
    return val_loss / len(test_loader)


# Run the optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Run 20 trials

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the best hyperparameters
    best_params = study.best_trial.params
    with open("models/best_hyperparameters.txt", "w") as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    print("Best hyperparameters saved!")
