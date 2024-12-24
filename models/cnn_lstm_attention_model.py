import torch
import torch.nn as nn

class CNNLSTMWithAttention(nn.Module):
    def __init__(self, input_size, cnn_filters, lstm_hidden_size, num_layers, output_size, dropout=0.2):
        super(CNNLSTMWithAttention, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Attention weights layer
        self.attention = nn.Linear(lstm_hidden_size, 1)

        # Fully connected output layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        # CNN expects input of shape (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.cnn(x)

        # Transpose for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Compute attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Compute weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layer
        output = self.fc(context_vector)
        return output
