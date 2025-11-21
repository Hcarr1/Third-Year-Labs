import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def generate_sine_data(seq_length=50, num_samples=1000):
    X, y = [], []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        xs = np.sin(np.linspace(start, start + seq_length * 0.1, seq_length + 1))
        X.append(xs[:-1])
        y.append(xs[1:])
    return np.array(X), np.array(y)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

def plots(X, y, sample, title):
    Xs = X[sample,:,:].cpu().squeeze().numpy()
    ys = y[sample,:,:].cpu().squeeze().numpy()
    plt.figure(figsize=(6,4))
    plt.plot(Xs, label='X', linewidth=2)
    plt.plot(ys, label='y', linestyle='--')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out