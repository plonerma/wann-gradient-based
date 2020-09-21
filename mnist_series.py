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

        yield epoch


def evaluate(model, data):
    with model.discrete():  # discretize weights in model

        # add another dimenstion for weights and expand input for the number
        # of sampled shared weights

        acc = 0
        
        for x, y in data:

            x = x.unsqueeze(dim=0).expand((model.shared_weight.size()[0], -1, -1))

            predict_out = model(x).view(-1, model.out_features)

            _, predict_y = torch.max(predict_out, 1)
            
            acc += 1/len(data) * accuracy_score(y.repeat(model.shared_weight.size()).data, predict_y.data)
        
        return acc


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
                   distribution_descr='', run_name='MNIST', lr=0.1,
                   batch_size=64, directory='tmp_series'):

    # initialize writer for logging data to tensorboard
    
    run_dir = get_vacant_file_name(os.path.join(directory, run_name))
    os.mkdir(run_dir)

    writer = SummaryWriter(f"{run_dir}/summary")

    logging.info("Starting experiment")
    logging.info(f"Writing data to {writer.log_dir}")

    if mnist_size == 256:
        mnist = mnist_256
    else:
        mnist = mnist_full

    logging.info("Loading data")
    
    data = mnist()
    test_data = mnist(test=True)

    logging.info("Building the model")

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
        'name': run_name,
        'STE': use_ste,
        'Alpha-Blending': use_alpha_blending,
    }

    logging.info("Parameters:")
    logging.info(tabulate(hparams.items(), tablefmt='grid'))

    logging.info("Starting training")

    start_time = time.time()

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
            logging.info(f"Completed {(epoch + 1)} epochs. Current acc: {acc}")

    elapsed_time = time.time() - start_time

    writer.add_hparams(hparams, dict(
        accuracy=acc,
    ))
    writer.close()

    logging.info("Saving model.")
    torch.save(model.to_dict(), os.path.join(run_dir, 'model.pt')

    with open(f"{run_dir}/layer_sizes.txt", 'w') as f:
        f.write('\n'.join(map(str, (
                    [model.in_features]
                    + list(model.layer_sizes().numpy())
                    + [model.out_features]
               ))))

    logging.info("Done.")
    
    res = dict(params)
    res['accuracy'] = acc
    res['update_mechanism'] = update_mechanism
    res['run_name'] = run_name
    res['distribution'] = dist_descr
    res['elapsed_time'] = elapsed_time
    
    with open(os.path.join(run_dir, 'params.toml'), 'w') as f:
        f.write(toml.dumps(res))
        
    return red


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

        dist_descr, dist_func = params.pop('distribution')
        rep = params.pop('repetitions')
        update_mechanism = params.pop('update_mechanism')

        run_name = "MNIST-{mnist_size}_{grow}-{update}_{epochs}epochs-{batch_size}_{lr}_{dist}-{num_weights}_{rep}".format(
            rep=rep, dist=dist_descr,
            update=update_mechanism,
            grow='growing' if params['layer_sizes'] is None else 'static',
            **params
        )

        logging.info(f"Run {run_name}")

        use_ste, use_alpha_blending = dict(
            ste=(True, False),
            ab=(False, True),
            hybrid=(True, True),
        )[update_mechanism]

        res = run_experiment(
            use_ste=use_ste,
            use_alpha_blending=use_alpha_blending,
            distribution=dist_func,
            distribution_descr=dist_descr,
            directory=directory,
            run_name=run_name,
            **params
            )

        
        

        data.append(res)

        pd.DataFrame(data).to_csv(os.path.join(directory, get_vacant_file_name('results.csv')))


if __name__ == '__main__':
    series = dict(
        path='mnist_study_2',
        num_weights=5,
        epochs=100,
        batch_size=64,
        distribution=(
            ('uniform', partial(dist_uniform, -2, 2)),
            ('constant', torch.ones),
            ('lognormal', partial(dist_lognormal, 0, 0.5))
        ),
        mnist_size=(16*16, 28*28),
        update_mechanism=('ste', 'ab', 'hybrid'),
        layer_sizes=(None, [64]*4),  # None -> growing model
        lr=1,
        repetitions=4,
        skip_experiments=0,
    )
    run_series(**series)
