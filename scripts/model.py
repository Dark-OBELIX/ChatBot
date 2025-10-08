# scripts/model.py
import torch
import torch.nn as nn

class ChatBotNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBotNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
