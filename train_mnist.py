import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score

from model import Model, write_hist

from tasks.mnist import mnist_256

import numpy as np


training_epochs = 100


x, y = mnist_256()
train_data = DataLoader(TensorDataset(torch.Tensor(x), torch.Tensor(y)), batch_size=1000)

test_x, test_y = mnist_256(test=True)

# add bias to the inputs

test_x = np.hstack([test_x, np.ones((test_x.shape[0], 1))])

test_X = Variable(torch.Tensor(test_x).float())
test_y = Variable(torch.Tensor(test_y).long())



writer = SummaryWriter(comment='_mnist')

criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss


def train(optimizer, model, epochs=100):
    print (f"Training for {epochs} epochs with {len(train_data)} batches")
    for epoch in range(epochs):

        for i, data in enumerate(train_data):
            x, y = data

            # add bias to the inputs
            x = np.hstack([x, np.ones((x.shape[0], 1))])

            x = Variable(torch.Tensor(x).float())
            y = Variable(torch.Tensor(y).long())

            # discretize weights in model
            model.discretize()

            optimizer.zero_grad()

            batch = x.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))
            out = model(batch).view(-1, model.n_out)
            loss = criterion(out, y.repeat(model.shared_weight.size()))

            loss.backward()

            # restore original weights
            model.restore()

            # update original weights
            optimizer.step()

            # clip weights
            model.clip()

            writer.add_scalar("Loss", loss.data, epoch * len(train_data) + i)

        write_hist(writer, model, epoch)
        acc = evaluate(model, epoch=epoch)

        print(f"Completed epoch #{epoch} with acc: {acc}")


def evaluate(model, epoch=0):
    # discretize weights in model
    model.discretize()

    # add another dimenstion for weights and expand input for the number
    # of sampled shared weights
    
    batch = test_X.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))

    predict_out = model(batch).view(-1, model.n_out)

    # restore original weights
    model.restore()

    _, predict_y = torch.max(predict_out, 1)

    acc = accuracy_score(test_y.repeat(model.shared_weight.size()).data, predict_y.data)

    writer.add_scalar('prediction accurary', acc, epoch)
    return acc


shared_weight = Variable(torch.Tensor([.8,.9,1.0,1.1,1.2]))

# 5 ins (including one bias), 3 hidden layers with 2 nodes each, 3 outputs

layer_sizes = (
    [257]  # inputs (including bias)
    + [10] * 8  # hidden layers
    + [10]) # outputs

model = Model(shared_weight, *layer_sizes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)

model.init_weights()

print(f"training: {training_epochs} epochs")

train(optimizer, model, training_epochs)
evaluate(model, epoch=training_epochs)
write_hist(writer, model, epoch=training_epochs)

print("Done.")
