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

from param_util import ParamParser, nested_update


def potential_paths(*fn):
    fn = os.path.join(*fn)

    yield fn

    for i in count():
        yield f'{fn}-{i}'


def vacant_path(*fn):
    for fn in potential_paths(*fn):
        if not os.path.exists(fn):
            return fn


def train(optimizer, model, data, epochs=100,
          use_ab=True, use_ste=False, grow=False, writer=None):
    n_weights = model.shared_weight.size()[0]
    criterion = torch.nn.CrossEntropyLoss()  # cross entropy loss

    for epoch in range(epochs):
        for i, (x, y) in enumerate(data):

            #logging.debug(f'batch {i} / {len(data)}')

            x = x.unsqueeze(dim=0).expand((n_weights, -1, -1))
            y = y.repeat(n_weights)

            optimizer.zero_grad()

            # alpha blending
            if use_ab:
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

            if writer:
                writer.add_scalar("Loss", loss.data, epoch * len(data) + i)
                writer.add_scalar("Training/Alpha", alpha, epoch * len(data) + i)

            if grow:
                with torch.no_grad():
                    # grow model if necessary
                    if model.grow():
                        # update params in optimizer

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


distributions = {
    'constant': torch.ones,
    'linspace': torch.linspace,
    'uniform': dist_uniform,
    'normal': dist_normal,
    'lognormal': dist_lognormal
}


def get_dist_descr(params):
    dist = params['distribution']

    if isinstance(dist, str):
        return dist
    elif 'params' not in dist:
        return dist['name']
    else:
        params = ', '.join([str(p) for p in dist['params']])
        return f"{dist['name']} ({params})"


def get_dist_func(params):
    dist = params['distribution']

    dist_params = list()

    if isinstance(dist, str):
        dist_name = dist
    else:
        dist_name = dist['name']
        if 'params' in dist:
            dist_params = dist['params']

    assert dist_name == 'constant' or len(dist_params) == 2

    return partial(distributions[dist_name], *dist_params)


def is_growing(params):
    model = params['model']
    return (
        (isinstance(model, str) and model.lower().strip() == 'growing')
        or ('growing' in model and model['growing']))


def get_layer_sizes(params):
    if is_growing(params):
        return None
    else:
        return params['model']['layer_sizes']


def run_experiment(**params):
    for run_dir in potential_paths(params['directory'], params['run_name']):
        if os.path.exists(run_dir):
            # check if experiment has been completed
            hp_path = os.path.join(run_dir, 'hparams.toml')
            if os.path.exists(hp_path):
                logging.info(f"Experiment already completed in '{run_dir}' - loading data.")
                with open(hp_path, 'r') as f:
                    return toml.load(f)
        else:
            break

    os.mkdir(run_dir)

    exp_log_path = os.path.join(run_dir, 'experiment.log')

    expHandler = logging.FileHandler(exp_log_path)
    expHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    expHandler.setFormatter(formatter)

    logging.root.addHandler(expHandler)

    logging.info(f"Starting experiment in {run_dir} "
                 f"(for full log see: {exp_log_path}).")



    use_ste, use_ab = dict(
        ste=(True, False),
        ab=(False, True),
        hybrid=(True, True),
    )[params['update_mechanism']]

    if params['mnist_size'] == 256:
        mnist = mnist_256
    else:
        mnist = mnist_full

    if params['write_summary']:
        # initialize writer for logging data to tensorboard
        writer = SummaryWriter(os.path.join(run_dir, 'summary'))
        logging.debug(f"Writing summary to {writer.log_dir}")
    else:
        writer = None

    logging.debug("Loading data")

    train_data = mnist(batch_size=params['batch_size'])
    test_data = mnist(test=True, batch_size=params['batch_size'])

    logging.debug("Building the model")

    dist_func = get_dist_func(params)

    # set shared weights
    def sample_weight(w):
        w.data = dist_func(params['num_weights'])

    shared_weight = Variable(torch.empty(params['num_weights']))

    sample_weight(shared_weight)

    model = Model(
        shared_weight, params['mnist_size'] + 1, 10, get_layer_sizes(params))
    model.init_weight()

    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

    hparams = dict(
        batches=len(train_data),
        growing=is_growing(params),
        distribution=get_dist_descr(params),
        use_ste=use_ste,
        use_ab=use_ab
    )

    hparams.update(params)

    logging.debug("params:")
    logging.debug(tabulate(hparams.items(), tablefmt='grid'))

    logging.debug("Starting training")

    start_time = time.time()

    for epoch in train(optimizer, model, data=train_data,
                       epochs=params['epochs'],
                       writer=writer, grow=is_growing(params),
                       use_ste=use_ste, use_ab=use_ab):

        logging.debug(f"Completed epoch {epoch}.")

        ls = model.layer_sizes()

        if writer:

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

        if (writer and epoch % 10 == 0) or epoch == params['epochs']:
            sample_weight(shared_weight)
            acc = float(evaluate(model, data=test_data))
            logging.debug(f"Completed {epoch } epochs. Current acc: {acc}")

            if writer:
                write_hist(writer, model, epoch)
                writer.add_scalar('Evaluation/Accurary on Test Data; discretized',
                                  acc, epoch)

    results = dict(
        accuracy=acc, elapsed_time=time.time() - start_time,
    )

    if writer:
        writer.add_hparams({
            k: v for k, v in hparams.items()
            if any([
                isinstance(v, t)
                for t in (int, float, str, bool, torch.Tensor)
            ])}, results)
        writer.close()

    hparams.update(results)

    logging.debug("Saving model.")
    torch.save(model.to_dict(), os.path.join(run_dir, 'model.pt'))

    with open(os.path.join(run_dir, 'hparams.toml'), 'w') as f:
        toml.dump(hparams, f)

    logging.info("Done.")

    logging.root.removeHandler(expHandler)

    return hparams


def get_const_var_params(params):
    const_params = dict()
    variable_params = dict()

    for k, v in params.items():
        if isinstance(v, list):
            variable_params[k] = v
        else:
            const_params[k] = v

    rep = params.get('repetitions', 1)
    if not isinstance(rep, list):
        rep = list(range(rep))

    variable_params['repetition'] = rep

    return const_params, variable_params

def get_num_total_experiments(variable_params):
    return reduce(
        lambda x, y: x*len(y), variable_params.values(), 1)

def run_series(**params):
    directory = params.pop('path')

    if not os.path.exists(directory):
        os.mkdir(directory)

    fileHandler = logging.FileHandler(vacant_path(directory, 'series.log'))
    fileHandler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fileHandler.setFormatter(formatter)

    logging.root.addHandler(fileHandler)

    df_path = vacant_path(directory, 'results.csv')

    logging.debug(f'Storing results in {df_path}.')

    start_at = params.pop('start_at', 0)
    stop_at = params.pop('stop_at', 0)

    assert isinstance(start_at, int) and isinstance(stop_at, int)

    const_params, variable_params = get_const_var_params(params)

    num_total_experiments = get_num_total_experiments(variable_params)

    if stop_at < 0:
        stop_at = num_total_experiments

    if stop_at > start_at:
        num_experiments_to_run = stop_at - start_at
    else:
        num_experiments_to_run = 0

    logging.info(
        f"Running {num_experiments_to_run} "
        f"of {num_total_experiments} experiments.")

    data = list()

    experiments = product(*variable_params.values())

    for experiment_index, var_values in enumerate(experiments):

        if experiment_index < (start_at or 0):
            continue

        if experiment_index >= stop_at:
            logging.debug(f'Stopping (since stop_at={stop_at}).')
            break

        logging.debug(
            f"Experiment {experiment_index+1}/{num_total_experiments}")

        params = dict()
        nested_update(params, const_params)
        nested_update(params, dict(zip(variable_params.keys(), var_values)))

        run_name = (
            "MNIST-{mnist_size}_"
            "{grow}-{update_mechanism}_"
            "{epochs}epochs-{batch_size}_"
            "{lr}_{dist}-{num_weights}_{repetition}"
        ).format(
            dist=get_dist_descr(params),
            grow='growing' if is_growing(params) else 'static',
            **params
        )

        logging.info(f"Run {run_name}")

        res = run_experiment(
            directory=directory,
            run_name=run_name,
            **params
            )

        data.append(res)

        pd.DataFrame(data).to_csv(df_path)


if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logging.root.addHandler(sh)

    parser = ParamParser()

    parser.add_argument(
        "--num_experiments", action='store_true'
    )

    # default params
    params = dict(
        path='series_data',
        num_weights=1,
        epochs=1,
        batch_size=1,
        distribution='constant',
        mnist_size=784,
        update_mechanism='ab',
        model='growing',
        lr=1,
        repetitions=1,
        start_at=0,
        stop_at=-1,
        write_summary=False,
    )

    args = parser.parse_args()
    parsed_params = args.params

    if len(parsed_params) == 0:
        logging.warning('No parameters defined. Using only default values.')

    nested_update(params, parsed_params)

    if args.num_experiments:
        _, var_params = get_const_var_params(params)
        print(get_num_total_experiments(var_params))
        quit(0)

    """series = dict(
        path='data/mnist_long_trial_2',
        num_weights=6,
        epochs=100,
        batch_size=64,
        distribution=(
            ('linspace', partial(dist_linspace, -2, 2)),
            ('uniform', partial(dist_uniform, -2, 2)),
            ('constant', torch.ones),
            ('lognormal', partial(dist_lognormal, 0, 1)),
        ),
        mnist_size=(16*16, 28*28),
        update_mechanism=('ste', 'ab', 'hybrid'),
        layer_sizes=(None, [64]*4),  # None -> growing model
        lr=(.5,1,2),
        repetitions=3,
        skip_experiments=54,
    )"""

    run_series(**params)
