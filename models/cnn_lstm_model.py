import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_size, cnn_filters, lstm_hidden_size, num_layers, output_size, dropout=0.2):
        super(CNNLSTM, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected output layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        # CNN expects input of shape (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.cnn(x)

        # Transpose for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)

        # LSTM forward pass
        x, _ = self.lstm(x)

        # Take the last output from LSTM
        x = x[:, -1, :]

        # Fully connected layer
        output = self.fc(x)
        return output
