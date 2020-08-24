from itertools import product
from functools import partial, reduce

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score

from model import Model, write_hist

from tasks.mnist import mnist_256, mnist_full

import numpy as np

from tabulate import tabulate


def train(optimizer, model, data, epochs=100,
          use_alpha_blending=True, use_ste=False, grow=False, writer=None):
    n_weights = model.shared_weight.size()[0]
    criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss

    for epoch in range(epochs):
        for i, (x, y) in enumerate(data):

            x = x.unsqueeze(dim=0).expand((n_weights, -1, -1))
            x = Variable(x)
            y = Variable(torch.Tensor(y).long().repeat(n_weights))

            optimizer.zero_grad()

            if use_alpha_blending:
                alpha = (epoch * len(data) + i) / (epochs * len(data))
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

            writer.add_scalar("Loss", loss.data, epoch * len(data) + i)
            writer.add_scalar("Training/Alpha", alpha, epoch * len(data) + i)

            if grow:
                with torch.no_grad():
                    # grow model if necessary
                    if model.grow():
                        # update parameters in optimizer

                        # This is ugly, but it seems to be the simplest way
                        optimizer.param_groups = list()
                        params = list(model.parameters())
                        optimizer.add_param_group({'params': params})

        yield epoch


def evaluate(model, data):
    with model.discrete():  # discretize weights in model

        # add another dimenstion for weights and expand input for the number
        # of sampled shared weights

        x, y = data

        x = x.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))

        predict_out = model(x).view(-1, model.out_features)

        _, predict_y = torch.max(predict_out, 1)

        return accuracy_score(y.repeat(model.shared_weight.size()).data,
                              predict_y.data)


def dist_uniform(a, b, size):
    sample = np.random.uniform(a, b, tuple(size)).astype('float32')
    return torch.from_numpy(sample)


def dist_normal(mu, sigma, size):
    sample = np.random.normal(mu, sigma, tuple(size)).astype('float32')
    return torch.from_numpy(sample)


def dist_lognormal(mu, sigma, size):
    sample = np.random.lognormal(mu, sigma, tuple(size)).astype('float32')
    return torch.from_numpy(sample)


def run_experiment(*, mnist_size=28*28, num_weights=10, layer_sizes=None,
                   epochs=400, use_ste=False, use_alpha_blending=True,
                   distribution=partial(dist_uniform, -2, 2),
                   distribution_descr='', model_file_name=None, lr=0.1,
                   batch_size=64):

    # initialize writer for logging data to tensorboard
    writer = SummaryWriter(comment='_mnist_{}'.format(str(mnist_size)))

    print("Starting experiment")

    if mnist_size == 256:
        mnist = mnist_256
    else:
        mnist = mnist_full

    print("Loading data")
    x, y = mnist()

    # add bias to the inputs
    x = np.hstack([x, np.ones((x.shape[0], 1))])

    train_data = DataLoader(
        TensorDataset(torch.Tensor(x).float(), torch.Tensor(y)),
        batch_size=batch_size, shuffle=True)

    test_x, test_y = mnist(test=True)

    # add bias to the test inputs
    test_x = np.hstack([test_x, np.ones((test_x.shape[0], 1))])

    test_data = Variable(torch.Tensor(test_x).float()), \
                Variable(torch.Tensor(test_y).long())

    print("Building the model")

    # set shared weights
    def sample_weight(w):
        w.data = distribution(w.size())

    shared_weight = Variable(torch.empty(num_weights))

    sample_weight(shared_weight)


    model = Model(shared_weight, mnist_size + 1, 10, layer_sizes)
    model.init_weight()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    growing = (layer_sizes is None)

    hparams = {
        'epochs': epochs,
        'batch size': batch_size,
        'batches': len(train_data),
        'learning rate': lr,
        'number of weights': num_weights,
        'MNIST': mnist_size,
        'growing': growing,
        'layer sizes': str(layer_sizes),
        'distribution': distribution_descr,
        'model file name': model_file_name,
        'STE': use_ste,
        'Alpha-Blending': use_alpha_blending,
    }

    print("Parameters:")
    print(tabulate(hparams.items(), tablefmt='grid'))

    print("Starting training")
    for epoch in train(optimizer, model, data=train_data, epochs=epochs,
                       writer=writer, grow=growing, use_ste=use_ste,
                       use_alpha_blending=use_alpha_blending):
        ls = model.layer_sizes()

        for label, s, in [
            ("Architecture/Num Weights", model.numel()),
            ("Architecture/Hidden Layers", len(model.hidden_layers)),
            ("Architecture/Biggest Layer", ls.max()),
            ("Architecture/Mean Layer", ls.mean()),
            ("Architecture/Num Nodes", ls.sum()),
            ("Architecture/Nodes without input",
             model.nodes_without_input()),
            ("Architecture/Output nodes without input",
             model.output_layer.nodes_without_input().shape[0]),
        ]:
            writer.add_scalar(label, s, epoch * len(train_data))

        if (epoch + 1) % 10 == 0:
            sample_weight(shared_weight)
            write_hist(writer, model, (epoch + 1))
            acc = evaluate(model, data=test_data)
            writer.add_scalar('Evaluation/Accurary on Test Data; discretized',
                              acc, (epoch + 1))
            print(f"Completed {(epoch + 1)} epochs. Current acc: {acc}")

    writer.add_hparams(hparams, dict(
        accuracy=acc,
    ))
    writer.close()

    if model_file_name is not None:
        print("Saving model.")
        torch.save(model.to_dict(), model_file_name)


    print("Done.")


def run_series(**parameters):
    model_file_name_template = 'models/' + '_'.join(
        ['mnist'] + [f'{{{x}}}' for x in [
            'mnist_size', 'grow_descr', 'update_mechanism', 'dist_descr', 'lr',
            'rep']]
        ) + '.pt'

    const_params = dict()
    variable_params = dict()

    parameters['repetitions'] = tuple(range(parameters.get('repetitions', 1)))

    for k, v in parameters.items():
        if isinstance(v, tuple):
            variable_params[k] = v
        else:
            const_params[k] = v

    num_experiments = reduce(lambda x, y: x*len(y), variable_params.values(), 1)
    print(f"Running series of {num_experiments} experiments.")

    for var_values in product(*variable_params.values()):
        params = dict(const_params)

        for k, v in zip(variable_params.keys(), var_values):
            params[k] = v

        dist_descr, dist_func = params.pop('distribution')
        rep = params.pop('repetitions')
        update_mechanism = params.pop('update_mechanism')

        model_file_name = model_file_name_template.format(
            rep=rep, dist_descr=dist_descr,
            update_mechanism=update_mechanism,
            grow_descr='growing' if params['layer_sizes'] is None else 'static',
            **params
        )

        print(f"Building model {model_file_name}")

        use_ste, use_alpha_blending = dict(
            ste=(True, False),
            ab=(False, True),
            hybrid=(True, True),
        )[update_mechanism]

        run_experiment(
            use_ste=use_ste,
            use_alpha_blending=use_alpha_blending,
            distribution=dist_func,
            distribution_descr=dist_descr,
            model_file_name=model_file_name,
            **params
            )


if __name__ == '__main__':
    series = dict(
        num_weights=6,
        epochs=100,
        distribution=(
        #    ('uniform', partial(dist_uniform, -2, 2)),
            ('constant', torch.ones),
        #    ('lognormal', partial(dist_lognormal, 0, 0.5))
        ),
        mnist_size=(28*28,),#(16*16, 28*28)
        update_mechanism=('ste', 'ab', 'hybrid'),
        layer_sizes=(None, [64]*4),  # None -> growing model
        lr=(0.01 ,0.05, 0.1, 0.2, 0.5, 1, 2),
        repetitions=1,
    )
    run_series(**series)
