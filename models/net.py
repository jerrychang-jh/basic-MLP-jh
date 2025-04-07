import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(10 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)

        return logits