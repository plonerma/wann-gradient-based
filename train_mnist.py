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

in_features = 16*16
out_features = 10

#layer_sizes = [10]*3
layer_sizes = None

num_weights = 10

grow = (layer_sizes is None)

model_file_name = 'tmp_model.pt'


use_alpha_blending = True
use_ste = False

distribution = 'lognormal', 0, 0.5

# prepare data

x, y = mnist()
train_data = DataLoader(TensorDataset(torch.Tensor(x), torch.Tensor(y)), batch_size=1000, shuffle=True)

test_x, test_y = mnist(test=True)

# add bias to the test inputs
test_x = np.hstack([test_x, np.ones((test_x.shape[0], 1))])

test_X = Variable(torch.Tensor(test_x).float())
test_y = Variable(torch.Tensor(test_y).long())


criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss


def sample_weight(w):
    np.random.lognormal()
    np.random.lognormal()
    dist, a, b = distribution

    r_func = dict(
        uniform=np.random.uniform,
        normal=np.random.normal,
        lognormal=np.random.lognormal,
    )

    w.data = torch.from_numpy(
        r_func[dist](a, b, tuple(w.size())).astype('float32')
    )


def train(optimizer, model, epochs=100):
    print(f"Training for {epochs} epochs with {len(train_data)} batches")
    for epoch in range(epochs):
        sample_weight(model.shared_weight)

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

            if use_alpha_blending:
                alpha = (epoch * len(train_data) + i) / (epochs * len(train_data))
            else:
                alpha = 1

            with model.alpha_blend(alpha):
                out = model(x).view(-1, model.out_features)
                loss = criterion(out, y)
                loss.backward()

            if not use_ste:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        p.grad = p.grad * (1-alpha)

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

            ls = model.layer_sizes()

            if grow:
                scalars += [
                    ("Architecture/Num Weights", model.numel()),
                    ("Architecture/Hidden Layers", len(model.hidden_layers)),
                    ("Architecture/Biggest Layer", ls.max()),
                    ("Architecture/Mean Layer", ls.mean()),
                    ("Architecture/Num Nodes", ls.sum()),
                    ("Architecture/Nodes without input", model.nodes_without_input()),
                    ("Architecture/Output nodes without input", model.output_layer.nodes_without_input().shape[0]),
                ]

            for label, s, in scalars:
                writer.add_scalar(label, s, epoch * len(train_data) + i)

        if epoch % 5 == 0:
            write_hist(writer, model, epoch)
            acc = evaluate(model, epoch=epoch)
            print(f"Completed epoch #{epoch} with acc: {acc}")


def evaluate(model, epoch=None):
    sample_weight(model.shared_weight)


    with model.discrete():  # discretize weights in model

        # add another dimenstion for weights and expand input for the number
        # of sampled shared weights

        x = test_X.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))

        predict_out = model(x).view(-1, model.out_features)

        _, predict_y = torch.max(predict_out, 1)

        acc = accuracy_score(test_y.repeat(model.shared_weight.size()).data, predict_y.data)

        if epoch is not None:
            writer.add_scalar('Evaluation/Accurary on Test Data; discretized', acc, epoch)

        return acc


if __name__ == "__main__":

    # initialize writer for logging data to tensorboard
    writer = SummaryWriter(comment='_mnist')

    # set shared weights
    shared_weight = Variable(torch.empty(num_weights))

    model = Model(shared_weight, in_features + 1, out_features, layer_sizes)
    model.init_weight()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    print(f"training: {training_epochs} epochs")

    train(optimizer, model, training_epochs)
    acc = evaluate(model, epoch=training_epochs)
    write_hist(writer, model, epoch=training_epochs)

    writer.add_hparams(dict(
        epochs=training_epochs,
        mnist_size=in_features,
        alpha_blending=use_alpha_blending,
        ste=use_ste,
        growing=grow,
        num_weights=num_weights,
        distribution='{} ({}, {})'.format(*distribution),
    ), dict(
        accuracy=acc,
    ))

    if model_file_name is not None:
        print("Saving model.")
        torch.save(model.to_dict(), model_file_name)

    print("Done.")
