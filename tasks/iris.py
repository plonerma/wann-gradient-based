from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np

import torch
from torch.autograd import Variable

def iris(test=False):
    dataset = load_iris()
    train_X, test_X, train_y, test_y = train_test_split(dataset['data'],
                                                        dataset['target'], test_size=0.2)

    if not test:
        return train_X, train_y
    else:
        return test_X, test_y
