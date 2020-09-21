from itertools import product, count
from functools import partial, reduce

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


from sklearn.metrics import accuracy_score

from model import Model, write_hist

from tasks.mnist import mnist_256, mnist_full

import numpy as np

from tabulate import tabulate

import os
import os.path

import time

import pandas as pd

import logging

import toml


def get_vacant_file_name(fn):
    i = None

    if os.path.exists(fn):
        for i in count():
            _fn = f'{fn}-{i}'
            if not os.path.exists(_fn):
                logging.warning(f"Filename {fn} already taken. Using {_fn} instead.")
                return _fn
    else:
        return fn


def train(optimizer, model, data, epochs=100,
          use_alpha_blending=True, use_ste=False, grow=False, writer=None):
    n_weights = model.shared_weight.size()[0]
    criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss

    for epoch in range(epochs):
        for i, (x, y) in enumerate(data):

            x = x.unsqueeze(dim=0).expand((n_weights, -1, -1))
            y = y.repeat(n_weights)

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
                with torch.no_grad():
                    for p in model.parameters():
                        p.grad.data = p.grad * (1-alpha)

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

        yield epoch + 1


def evaluate(model, data):
    with model.discrete():  # discretize weights in model

        # add another dimenstion for weights and expand input for the number
        # of sampled shared weights

        acc = 0

        n_weights = model.shared_weight.size()[0]

        for x, y in data:

            x = x.unsqueeze(dim=0).expand((n_weights, -1, -1))
            y = y.repeat(n_weights)

            predict_out = model(x).view(-1, model.out_features)
            _, predict_y = torch.max(predict_out, 1)

            acc += 1/len(data) * accuracy_score(y.data, predict_y.data)

        return acc


def dist_uniform(a, b, size):
    sample = np.random.uniform(a, b, int(size)).astype('float32')
    return torch.from_numpy(sample)


def dist_normal(mu, sigma, size):
    sample = np.random.normal(mu, sigma, int(size)).astype('float32')
    return torch.from_numpy(sample)


def dist_lognormal(mu, sigma, size):
    sample = np.random.lognormal(mu, sigma, int(size)).astype('float32')
    return torch.from_numpy(sample)


def dist_linspace(a, b, size):
    return torch.linspace(a, b, int(size))


defaults = dict(
    num_weights=10, layer_sizes=None,
    epochs=400, lr=0.1, batch_size=64,
    use_ste=False, use_alpha_blending=True,
    distribution=('uniform', partial(dist_uniform, -2, 2)),
    run_name='MNIST', directory='tmp_series'
)


def run_experiment(**params):
    d = dict(defaults)
    d.update(params)

    # initialize writer for logging data to tensorboard

    run_dir = get_vacant_file_name(os.path.join(d['directory'], d['run_name']))
    os.mkdir(run_dir)

    use_ste, use_alpha_blending = dict(
        ste=(True, False),
        ab=(False, True),
        hybrid=(True, True),
    )[d['update_mechanism']]

    if d['mnist_size'] == 256:
        mnist = mnist_256
    else:
        mnist = mnist_full

    dist_descr, dist_func = d['distribution']

    writer = SummaryWriter(os.path.join(run_dir, 'summary'))

    logging.info("Starting experiment")
    logging.info(f"Writing data to {writer.log_dir}")

    logging.info("Loading data")

    train_data = mnist()
    test_data = mnist(test=True)

    logging.info("Building the model")

    # set shared weights
    def sample_weight(w):
        w.data = dist_func(d['num_weights'])

    shared_weight = Variable(torch.empty(d['num_weights']))

    sample_weight(shared_weight)

    model = Model(shared_weight, d['mnist_size'] + 1, 10, d['layer_sizes'])
    model.init_weight()

    optimizer = torch.optim.SGD(model.parameters(), lr=d['lr'])

    growing = (d['layer_sizes'] is None)

    hparams = dict(d)
    hparams['batches'] = len(train_data)
    hparams['growing'] = growing
    hparams['distribution'] = dist_descr
    hparams['use_ste'] = use_ste
    hparams['use_ab'] = use_alpha_blending

    logging.info("Parameters:")
    logging.info(tabulate(hparams.items(), tablefmt='grid'))

    logging.info("Starting training")

    start_time = time.time()

    for epoch in train(optimizer, model, data=train_data, epochs=d['epochs'],
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

        if epoch % 10 == 0 or epoch == d['epochs']:
            sample_weight(shared_weight)
            write_hist(writer, model, epoch)
            acc = evaluate(model, data=test_data)
            writer.add_scalar('Evaluation/Accurary on Test Data; discretized',
                              acc, epoch)
            logging.info(f"Completed {epoch } epochs. Current acc: {acc}")

    results = dict(
        accuracy=acc, elapsed_time=time.time() - start_time,
    )

    writer.add_hparams({k:v for k,v in hparams.items() if any([isinstance(v, t) for t in (int, float, str, bool, torch.Tensor)])}, results)
    writer.close()

    hparams.update(results)

    logging.info("Saving model.")
    torch.save(model.to_dict(), os.path.join(run_dir, 'model.pt'))

    with open(os.path.join(run_dir, 'hparams.toml'), 'w') as f:
        f.write(toml.dumps(hparams))

    logging.info("Done.")

    return hparams


def run_series(**parameters):
    const_params = dict()
    variable_params = dict()

    directory = parameters.pop('path')

    if not os.path.exists(directory):
        os.mkdir(directory)

    logging.getLogger().setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(os.path.join(directory, 'series.log'))
    fileHandler.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    logging.getLogger().addHandler(fileHandler)
    logging.getLogger().addHandler(consoleHandler)

    parameters['repetitions'] = tuple(range(parameters.get('repetitions', 1)))
    skip_experiments = parameters.pop('skip_experiments', 0)

    for k, v in parameters.items():
        if isinstance(v, tuple):
            variable_params[k] = v
        else:
            const_params[k] = v

    num_experiments = reduce(lambda x, y: x*len(y), variable_params.values(), 1)
    logging.info(f"Running series of {num_experiments} experiments.")

    data = list()

    for experiment_index, var_values in enumerate(product(*variable_params.values())):
        params = dict(const_params)

        logging.debug(f"Experiment {experiment_index+1}/{num_experiments}")

        if experiment_index < (skip_experiments or 0):
            logging.debug("Skipping experiment")
            continue

        for k, v in zip(variable_params.keys(), var_values):
            params[k] = v

        params['repetition'] = params.pop('repetitions')

        run_name = "MNIST-{mnist_size}_{grow}-{update_mechanism}_{epochs}epochs-{batch_size}_{lr}_{dist}-{num_weights}_{repetition}".format(
            dist=params['distribution'][0],
            grow='growing' if params['layer_sizes'] is None else 'static',
            **params
        )

        logging.info(f"Run {run_name}")

        res = run_experiment(
            directory=directory,
            run_name=run_name,
            **params
            )

        data.append(res)

        pd.DataFrame(data).to_csv(os.path.join(directory, get_vacant_file_name('results.csv')))


if __name__ == '__main__':
    series = dict(
        path='data/mnist_uniform',
        num_weights=10,
        epochs=100,
        batch_size=64,
        distribution=(
            ('linspace', partial(dist_linspace, -2, 2)),
        ),
        mnist_size=(16*16, 28*28),
        update_mechanism='hybrid',#('ste', 'ab', 'hybrid'),
        layer_sizes=(None, [64]*4),  # None -> growing model
        lr=1,
        repetitions=4,
        skip_experiments=0,
    )
    run_series(**series)
