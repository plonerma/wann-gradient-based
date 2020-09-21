from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np

from .util import prepare_data

import torch
from torch.autograd import Variable

def iris(test=False, **kw):
    dataset = load_iris()
    train_X, test_X, train_y, test_y = train_test_split(dataset['data'],
                                                        dataset['target'], test_size=0.2)

    if not test:
        return prepare_data(train_X, train_y, **kw)
    else:
        return prepare_data(test_X, test_y, **kw)
