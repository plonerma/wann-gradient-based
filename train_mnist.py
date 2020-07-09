import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score

from model import Model, write_hist

from tasks.mnist import mnist_256 as mnist

import numpy as np




# parameters

training_epochs = 400
use_weight_alpha_blending = True
use_loss_alpha_blending = False

in_features = 256
out_features = 10

#layer_sizes = [10]*3
layer_sizes = None

grow = (layer_sizes is None)


# prepare data

x, y = mnist()
train_data = DataLoader(TensorDataset(torch.Tensor(x), torch.Tensor(y)), batch_size=1000, shuffle=True)

test_x, test_y = mnist(test=True)

# add bias to the test inputs
test_x = np.hstack([test_x, np.ones((test_x.shape[0], 1))])

test_X = Variable(torch.Tensor(test_x).float())
test_y = Variable(torch.Tensor(test_y).long())


# initialize writer for logging data to tensorboard
writer = SummaryWriter(comment='_mnist_static')


# set shared weights
shared_weight = Variable(torch.Tensor([.8,.9,1.0,1.1,1.2]))



criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss

def train(optimizer, model, epochs=100):
    print (f"Training for {epochs} epochs with {len(train_data)} batches")
    for epoch in range(epochs):

        print(f"#{epoch}")

        for i, data in enumerate(train_data):
            x, y = data

            n_weights = model.shared_weight.size()[0]

            # add bias to the inputs
            x = np.hstack([x, np.ones((x.shape[0], 1))])
            x = torch.Tensor(x).float()
            x = x.unsqueeze(dim=0).expand((n_weights, -1, -1))
            x = Variable(x)
            y = Variable(torch.Tensor(y).long().repeat(n_weights))

            optimizer.zero_grad()

            alpha = (epoch * len(train_data) + i) / (epochs * len(train_data))

            if use_loss_alpha_blending:
                out = model(x).view(-1, model.out_features)
                loss_continous = criterion(out, y)

            weight_alpha = (alpha if use_weight_alpha_blending else 1)
            with model.discrete(alpha=weight_alpha):
                out = model(x).view(-1, model.out_features)
                loss_discrete = criterion(out, y)

                loss = loss_discrete

                if use_loss_alpha_blending:
                    loss *= alpha
                    loss += (1-alpha) + loss_continous

                loss.backward()

            # update original weights
            optimizer.step()

            # clip weights
            model.clip()

            if grow:
                with torch.no_grad():
                    # grow model if necessary
                    if model.grow():
                        # update parameters in optimizer

                        # This is ugly, but it seems to be the simplest way
                        optimizer.param_groups = list()
                        params = list(model.parameters())
                        optimizer.add_param_group({'params': params})


            scalars = [
                ("Loss", loss.data),
                ("Training/Alpha", alpha),
            ]

            if use_loss_alpha_blending:
                scalars += [
                    ("Loss/Discrete", loss_discrete.data),
                    ("Loss/Continous", loss_continous.data),
                ]

            ls = model.layer_sizes()

            if grow:
                scalars += [
                    ("Architecture/Num Weights", model.numel()),
                    ("Architecture/Hidden Layers", len(model.hidden_layers)),
                    ("Architecture/Biggest Layer", ls.max()),
                    ("Architecture/Mean Layer", ls.mean()),
                    ("Architecture/Num Nodes", ls.sum()),
                    ("Architecture/Nodes without input", model.nodes_without_input())
                ]

            for label, s, in scalars:
                writer.add_scalar(label, s, epoch * len(train_data) + i)

        if epoch % 5 == 0:
            write_hist(writer, model, epoch)
            acc = evaluate(model, epoch=epoch)
            print(f"Completed epoch #{epoch} with acc: {acc}")


def evaluate(model, epoch=0):
    with model.discrete():  # discretize weights in model

        # add another dimenstion for weights and expand input for the number
        # of sampled shared weights

        x = test_X.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))

        predict_out = model(x).view(-1, model.out_features)

        _, predict_y = torch.max(predict_out, 1)

        acc = accuracy_score(test_y.repeat(model.shared_weight.size()).data, predict_y.data)

        writer.add_scalar('Evaluation/Accurary on Test Data; discretized', acc, epoch)
        return acc




model = Model(shared_weight,
              in_features + 1, out_features,
              layer_sizes, grow=grow)

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

model.init_weights()

print(f"training: {training_epochs} epochs")

train(optimizer, model, training_epochs)
evaluate(model, epoch=training_epochs)
write_hist(writer, model, epoch=training_epochs)

print("Done.")
