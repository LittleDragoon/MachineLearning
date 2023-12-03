import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


# download data from https://archive.ics.uci.edu/ml/datasets/spambase
FILE_NAME = "spambase.data"


# Loads the CSV data
df = pd.read_csv(FILE_NAME, header=None)

# The first 57 columns are features & the last column has the correct labels (targets)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

# Standardize the features
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


# Manual split into train and test sets
SPLIT_RATIO = 0.8
split_idx = int(SPLIT_RATIO * len(X))

X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Create a PyTorch dataset and data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Define the LSTM model
class SpamLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=True):
        super(SpamLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out) 
        out = self.sigmoid(out)
        return out

# Initialize the LSTM model
INPUT_SIZE = X.shape[1]
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

lstm_model = SpamLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

loss_values = []  # To store loss values

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    lstm_model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = lstm_model(batch_X)
        loss = criterion(outputs, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print(loss_values)
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.tight_layout()
plt.show()
