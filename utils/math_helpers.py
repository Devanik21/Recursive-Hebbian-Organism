import numpy as np

def normalize(weights):
    return weights / (np.linalg.norm(weights) + 1e-8)
