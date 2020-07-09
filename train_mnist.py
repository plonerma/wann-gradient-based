import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score

from model import Model, write_hist

from tasks.mnist import mnist_256

import numpy as np




# PARAMETERS

training_epochs = 400
use_weight_alpha_blending = True
use_loss_alpha_blending = False







x, y = mnist_256()
train_data = DataLoader(TensorDataset(torch.Tensor(x), torch.Tensor(y)), batch_size=1000, shuffle=True)

test_x, test_y = mnist_256(test=True)

# add bias to the inputs

test_x = np.hstack([test_x, np.ones((test_x.shape[0], 1))])

test_X = Variable(torch.Tensor(test_x).float())
test_y = Variable(torch.Tensor(test_y).long())



writer = SummaryWriter(comment='_mnist_static')

criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss


def train(optimizer, model, epochs=100):
    print (f"Training for {epochs} epochs with {len(train_data)} batches")
    for epoch in range(epochs):

        for i, data in enumerate(train_data):
            x, y = data

            n_weights = model.shared_weight.size()[0]

            # add bias to the inputs
            x = np.hstack([x, np.ones((x.shape[0], 1))])

            x = Variable(torch.Tensor(x).float().unsqueeze(dim=0).expand((n_weights, -1, -1)))
            y = Variable(torch.Tensor(y).long().repeat(n_weights))

            # discretize weights in model
            alpha = (epoch * len(train_data) + i) / (epochs * len(train_data))

            if use_loss_alpha_blending:
                out = model(x).view(-1, model.out_features)
                loss_continous = criterion(out, y)


            model.discretize(alpha=(alpha if use_weight_alpha_blending else 1))

            out = model(x).view(-1, model.out_features)
            loss_discrete = criterion(out, y)

            if use_loss_alpha_blending:
                loss = loss_discrete * alpha + loss_continous * (1-alpha)
            else:
                loss = loss_discrete

            optimizer.zero_grad()

            loss.backward()

            # restore original weights
            model.restore()

            # update original weights
            optimizer.step()

            # clip weights
            model.clip()

            scalars = [
                ("Loss", loss.data),
                ("Training/Alpha", alpha),
            ]

            if use_loss_alpha_blending:
                scalars += [
                    ("Loss/Discrete", loss_discrete.data),
                    ("Loss/Continous", loss_continous.data),
                ]

            for label, s, in scalars:
                writer.add_scalar(label, s, epoch * len(train_data) + i)

        write_hist(writer, model, epoch)
        acc = evaluate(model, epoch=epoch)

        print(f"Completed epoch #{epoch} with acc: {acc}")


def evaluate(model, epoch=0):
    # discretize weights in model
    model.discretize()

    # add another dimenstion for weights and expand input for the number
    # of sampled shared weights

    batch = test_X.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))

    predict_out = model(batch).view(-1, model.out_features)

    # restore original weights
    model.restore()

    _, predict_y = torch.max(predict_out, 1)

    acc = accuracy_score(test_y.repeat(model.shared_weight.size()).data, predict_y.data)

    writer.add_scalar('Evaluation/Accurary on Test Data; discretized', acc, epoch)
    return acc


shared_weight = Variable(torch.Tensor([.8,.9,1.0,1.1,1.2]))

# 5 ins (including one bias), 3 hidden layers with 2 nodes each, 3 outputs

layer_sizes = (
    [257]  # inputs (including bias)
    + [10] * 3  # hidden layers
    + [10]) # outputs

model = Model(shared_weight, *layer_sizes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

model.init_weights()

print(f"training: {training_epochs} epochs")

train(optimizer, model, training_epochs)
evaluate(model, epoch=training_epochs)
write_hist(writer, model, epoch=training_epochs)

print("Done.")
