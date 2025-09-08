import numpy as np

def onehot(dim, index):
    """
    One-hot encoding of the index
    """
    onehot = np.zeros(dim, dtype=np.float32)
    onehot[index] = 1
    return onehot
