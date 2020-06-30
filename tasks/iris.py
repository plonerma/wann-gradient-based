from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np

import torch
from torch.autograd import Variable


dataset = load_iris()
#train_X, test_X, train_y, test_y = train_test_split(dataset['data'],
#                                                    dataset['target'], test_size=0.2)


# Just going for fitting here. (Todo: Change stuff in paper for usefull comparison)
train_X = dataset['data']
train_y = dataset['target']
test_X = train_X
test_y = train_y

# add bias to the inputs
train_X = np.hstack([train_X, np.ones((train_X.shape[0], 1))])
test_X = np.hstack([test_X, np.ones((test_X.shape[0], 1))])

# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())
