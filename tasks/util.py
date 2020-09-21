import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

def prepare_data(x, y, batch_size=None):
    # add bias to the inputs
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    
    x, y = torch.Tensor(x).float(), torch.Tensor(y).long()
    
    if batch_size is None:
        batch_size = x.shape[0]
    
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
