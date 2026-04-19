import torch

def normalize(weights):
    return weights / (torch.linalg.norm(weights) + 1e-8)
