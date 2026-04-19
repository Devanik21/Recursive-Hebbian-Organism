import torch
import torch.nn as nn

class PlasticityNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, pre, post, w):
        return self.layers(torch.stack([pre, post, w], dim=-1)).squeeze(-1)
