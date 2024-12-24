import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from models.cnn_lstm_model import CNNLSTM
from scripts.prepare_data import prepare_data

# Load prepared data
#cleaned_data_path = "C:/Users/gupta_k72mbnp/OneDrive/Documents/GitHub/Energy_Demand_Forecasting_System/data/processed/cleaned_energy_data.csv"
cleaned_data_path = "data/processed/cleaned_energy_data.csv"
sequence_length = 24
X_train, X_test, y_train, y_test, scaler = prepare_data(cleaned_data_path, sequence_length)

# Convert data to PyTorch tensors
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

# Initialize model
input_size = 1
cnn_filters = 16
lstm_hidden_size = 32
num_layers = 2
output_size = 1
dropout = 0.2

model = CNNLSTM(input_size, cnn_filters, lstm_hidden_size, num_layers, output_size, dropout)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_batch).squeeze()

        # Compute loss
        loss = criterion(y_pred, y_batch)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(test_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "models/cnn_lstm_model.pth")
print("Model saved successfully!")
