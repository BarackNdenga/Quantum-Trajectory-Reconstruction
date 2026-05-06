import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List

class TrajectoryNet(nn.Module):
    """
    Simple RNN-based model to predict true position from noisy measurements.
    """
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(TrajectoryNet, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

class NeuralInferenceEngine:
    """
    Wrapper for training and using a neural network for trajectory reconstruction.
    """
    def __init__(self):
        self.model = TrajectoryNet()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train_on_simulation(self, measurements: np.ndarray, true_positions: np.ndarray, epochs: int = 50):
        """
        Trains the model on a single trajectory.
        """
        self.model.train()
        
        # Prepare data
        x = torch.tensor(measurements, dtype=torch.float32).view(1, -1, 1)
        y = torch.tensor(true_positions, dtype=torch.float32).view(1, -1, 1)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Neural Training Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    def predict(self, measurements: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(measurements, dtype=torch.float32).view(1, -1, 1)
            prediction = self.model(x).numpy().flatten()
        return prediction
