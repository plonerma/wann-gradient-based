import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score

from growing_model import GrowingModel, write_hist

from tasks.mnist import mnist_256

import numpy as np


training_epochs = 200


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
    print(f"Training for {epochs} epochs with {len(train_data)} batches")

    for epoch in range(epochs):

        for i, data in enumerate(train_data):
            x, y = data

            n_weights = model.shared_weight.size()[0]

            # add bias to the inputs
            x = np.hstack([x, np.ones((x.shape[0], 1))])

            x = Variable(torch.Tensor(x).float().unsqueeze(dim=0).expand((n_weights, -1, -1)))
            y = Variable(torch.Tensor(y).long().repeat(n_weights))

            optimizer.zero_grad()

            out = model(x).view(-1, model.out_features)
            loss_continous = criterion(out, y)

            # discretize weights in model
            model.discretize()

            out = model(x).view(-1, model.out_features)
            loss_discrete = criterion(out, y)

            alpha = (epoch * len(train_data) + i) / (epochs * len(train_data))
            loss = alpha * loss_discrete + (1-alpha) * loss_continous

            loss.backward()

            # restore original weights
            model.restore()

            # update original weights
            optimizer.step()

            # clip weights
            model.clip()

            with torch.no_grad():
                # grow model if necessary
                if model.grow():
                    # update parameters in optimizer

                    # This is ugly, but it seems to be the simplest way
                    optimizer.param_groups = list()
                    params = list(model.parameters())
                    print(len(params))
                    optimizer.add_param_group({'params': params})

            layer_sizes = model.layer_sizes()

            scalars = [
                ("Loss/Combined", loss.data),
                ("Loss/Discrete", loss_discrete.data),
                ("Loss/Continous", loss_continous.data),
                ("Architecture/Num Weights", model.numel()),
                ("Architecture/Hidden Layers", len(model.hidden_layers)),
                ("Architecture/Biggest Layer", layer_sizes.max()),
                ("Architecture/Mean Layer", layer_sizes.mean()),
                ("Architecture/Num Nodes", layer_sizes.sum()),
                ("Architecture/Nodes without input", model.nodes_without_input()),
                ("Training/Alpha", alpha),
            ]
            for label, s, in scalars:
                writer.add_scalar(label, s, epoch * len(train_data) + i)

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


shared_weight = Variable(torch.Tensor([.8, .9, 1.0, 1.1, 1.2]))

model = GrowingModel(shared_weight, 257, 10)

# momentum cannot be used, since parameters might change
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

model.init_weights()

print(f"training: {training_epochs} epochs")

train(optimizer, model, training_epochs)
evaluate(model, epoch=training_epochs)
write_hist(writer, model, epoch=training_epochs)

print("Done.")
