"""This module implements a WANN-like network and straight through estimation
(STE, see @binnary_connect).

"""


import torch
from functools import reduce, partial
import numpy as np


class Discretizable:
    _is_discrete = False

    @property
    def is_discrete(self):
        return self._is_discrete

    def discretize_weight(self):
        if not self.is_discrete:
            self.stored_weight.data.copy_(self.weight.data)
            self.weight.data.copy_(self.effective_weight)
            self._is_discrete = True

    def restore_weight(self):
        if self.is_discrete:
            self.weight.data.copy_(self.stored_weight.data)
            self._is_discrete = False

    def clip_weight(self):
        torch.nn.functional.hardtanh(self.weight.data, inplace=True)



class MultiActivationModule(torch.nn.Module, Discretizable):
    """Applies multiple elementwise activation functions to a tensor."""

    available_act_functions = [
        ('relu', torch.relu),
        ('sigmoid', torch.sigmoid),
        ('tanh', torch.tanh),
        ('gaussian (standard)', lambda x: torch.exp(-torch.square(x) / 2.0)),
        ('step', lambda t: (t > 0.0) * 1.0),
        #('identity', lambda x: x),
        #('inverse', torch.neg),
        #('squared', torch.square),
        #('abs', torch.abs),
        #('cos', torch.cos),
        #('sin', torch.sin),
    ]

    def __init__(self, n_out):
        super().__init__()
        self.funcs = [f[1] for f in self.available_act_functions]

        self.out_features = n_out

        self.weight = torch.nn.Parameter(torch.zeros((self.n_funcs, n_out)))
        self.stored_weight = torch.empty_like(self.weight)

    @property
    def n_funcs(self):
        return len(self.funcs)

    def init_weight(self):
        torch.nn.init.uniform_(self.weight.data, 0, 1)
        self.clip_weight()

    def clip_weight(self):
        length = torch.norm(self.weight.data, dim=0).unsqueeze(dim=0)
        self.weight.data = self.weight.data / length

    @property
    def effective_weight(self):
        indices = torch.max(self.weight, 0).indices
        return torch.nn.functional.one_hot(indices, self.n_funcs).T.float()

    def forward(self, x):
        coefficients = self.weight

        return reduce(
            lambda first, act: (
                torch.add(
                    first,
                    torch.mul(
                        act[1](x),  # apply activation func
                        coefficients[act[0], :])
            )),
            enumerate(self.funcs),  # index, func
            torch.zeros_like(x)  # start value
        )


class TertiaryLinear(torch.nn.Linear, Discretizable):
    """Similar to torch.nn.Linear, with tertiary weights ($\in \{-1,0,1\}$)."""

    lambd = 0.4

    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out, bias=False)

        self.stored_weight = torch.empty_like(self.weight)

    @property
    def effective_weight(self):
        return torch.sign(torch.nn.functional.hardshrink(self.weight, lambd=self.lambd))

    def init_weight(self):
        torch.nn.init.normal_(self.weight.data, std=.12)


class ConcatLayer(torch.nn.Module):
    """Contatenates output of the active nodes and prior nodes."""

    def __init__(self, n_in, n_out, shared_weight):
        super().__init__()
        self.linear = TertiaryLinear(n_in, n_out)
        self.activation = MultiActivationModule(n_out)

        self.shared_weight = shared_weight

    def forward(self, x):
        linear = self.linear(x) * self.shared_weight[:, None, None]

        inner_out = self.activation(linear)

        return torch.cat([x, inner_out], dim=-1)


def discretize_weight(m):
    if hasattr(m, 'discretize_weight'):
        m.discretize_weight()


def restore_weight(m):
    if hasattr(m, 'restore_weight'):
        m.restore_weight()


def clip_weight(m):
    if hasattr(m, 'clip_weight'):
        m.clip_weight()


def weight_init(m):
    """Initialize weights randomly."""
    if isinstance(m, TertiaryLinear) or isinstance(m, MultiActivationModule):
        m.init_weight()


class Model(torch.nn.Module):
    def __init__(self, shared_weight, *layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes

        layers = list()

        n_in = layer_sizes[0]

        self.shared_weight = shared_weight

        for n_out in layer_sizes[1:]:
            layers.append(ConcatLayer(n_in, n_out, shared_weight))
            n_in += n_out

        self.network = torch.nn.Sequential(*layers)
        self.softmax = torch.nn.Softmax(dim=-1)

    def discretize(self):
        self.apply(discretize_weight)

    def restore(self):
        self.apply(restore_weight)

    def clip(self):
        self.apply(clip_weight)

    def init_weights(self):
        self.apply(weight_init)

    @property
    def in_features(self):
        return self.layer_sizes[0]

    @property
    def out_features(self):
        return self.layer_sizes[-1]

    def forward(self, x):
        net_out = self.network(x)
        net_out = net_out[..., -self.out_features:]
        return self.softmax(net_out)


def write_hist(writer, model, epoch):
    effective_weights = list()
    actual_weights = list()

    for m in model.modules():
        if isinstance(m, TertiaryLinear):
            effective_weights.append(m.effective_weight.reshape(-1))
            actual_weights.append(m.weight.reshape(-1))

    writer.add_histogram("effective weights", torch.cat(effective_weights), epoch)
    writer.add_histogram("actual weights", torch.cat(actual_weights), epoch)
