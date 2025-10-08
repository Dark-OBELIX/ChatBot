# scripts/model.py
import torch
import torch.nn as nn

class ChatBotMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(ChatBotMLP, self).__init__()

        layers = []
        prev_size = input_size

        # Création dynamique des couches cachées
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Couche de sortie
        layers.append(nn.Linear(prev_size, output_size))

        # Réseau complet
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
