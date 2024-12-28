import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from models.cnn_lstm_model import CNNLSTM
from models.cnn_lstm_attention_model import CNNLSTMWithAttention
from scripts.prepare_data import prepare_data

# Path to the saved model
#model_path = "models/cnn_lstm_model.pth"
model_path = "models/cnn_lstm_model.pth"

# Path to cleaned data
cleaned_data_path = "data/processed/cleaned_energy_data.csv"

# Prepare data for evaluation
sequence_length = 24
X_train, X_test, y_train, y_test, scaler = prepare_data(cleaned_data_path, sequence_length)

# Convert test data to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Initialize the model (use the same parameters as in training)
input_size = 1
cnn_filters = 64
lstm_hidden_size = 112
num_layers = 2
output_size = 1
dropout = 0.5
learning_rate = 0.0019512053461741875

model = CNNLSTM(input_size, cnn_filters, lstm_hidden_size, num_layers, output_size, dropout)

# Load the saved model weights
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Predict on test data
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy()

# Denormalize predictions and actual values
y_pred_denormalized = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_denormalized = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Calculate MAE and MSE
mae = np.mean(np.abs(y_pred_denormalized - y_test_denormalized))
mse = np.mean((y_pred_denormalized - y_test_denormalized)**2)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Visualize predictions vs. actuals
plt.figure(figsize=(12, 6))
plt.plot(y_test_denormalized[:100], label="Actual", marker="o")
plt.plot(y_pred_denormalized[:100], label="Predicted", marker="x")
plt.title("Actual vs. Predicted Energy Demand (Denormalized)")
plt.xlabel("Sample Index")
plt.ylabel("Energy Demand (Megawatthours)")
plt.legend()
plt.grid()
plt.show()
