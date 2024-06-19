import torch
import torch.nn as nn
import torch.optim as optim

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))

class NeuralNetworkWrapper:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def train(self, examples):
        self.model.train()
        for state, value in examples:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            value = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, value)
            loss.backward()
            self.optimizer.step()

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.model(state).item()
