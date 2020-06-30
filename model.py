import torch
from functools import reduce, partial
import numpy as np

class MultiActivationModule(torch.nn.Module):
    """Applies multiple elementwise activation functions to a tensor."""

    discretized = False

    available_act_functions = [
        ('relu', torch.relu),
        ('sigmoid', torch.sigmoid),
        ('tanh', torch.tanh),
        ('gaussian (standard)', lambda x: torch.exp(-torch.square(x) / 2.0)),
        ('step', lambda t: (t > 0.0) * 1.0),
        ('identity', lambda x: x),
        ('inverse', torch.neg),
        ('squared', torch.square),
        ('abs', torch.abs),
        ('cos', torch.cos),
        ('sin', torch.sin),
    ]

    @property
    def n_funcs(self):
        return len(self.funcs)

    def __init__(self, n_out):
        super().__init__()
        self.funcs = [f[1] for f in self.available_act_functions]

        self.weight = torch.nn.Parameter(torch.zeros((self.n_funcs, n_out)))
        self.frozen = torch.zeros(n_out, dtype=bool)
        self.soft = torch.nn.Softmax(dim=0)

    def forward(self, x):
        coefficients = self.soft(self.weight)

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


class RestrictedLinear(torch.nn.Module):
    """Similar to torch.nn.Linear, but restricts weights between -1 and 1."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.sign = torch.nn.Softsign()
        self.weight = torch.nn.Parameter(torch.empty(n_in, n_out))
        self.frozen = torch.zeros(*self.weight.size(), dtype=bool)

    def forward(self, x):
        weight = torch.where(self.frozen, self.weight, self.sign(self.weight) * 3/2)
        return torch.nn.functional.linear(x, weight.T)


class ConcatLayer(torch.nn.Module):
    """Contatenates output of the active nodes and prior nodes."""

    def __init__(self, n_in, n_out, shared_weight):
        super().__init__()
        self.linear = RestrictedLinear(n_in, n_out)
        self.activation = MultiActivationModule(n_out)

        self.shared_weight = shared_weight

    def forward(self, x):
        linear = self.linear(x) * self.shared_weight[:, None, None]

        inner_out = self.activation(linear)

        return torch.cat([x, inner_out], dim=-1)


def weight_init(m):
    """Initialize weights randomly."""
    if isinstance(m, RestrictedLinear):
        torch.nn.init.normal_(m.weight.data)
    elif isinstance(m, MultiActivationModule):
        torch.nn.init.normal_(m.weight.data)


def unfreeze_parameters(m):
    """Set all parameters to not frozen."""
    if isinstance(m, MultiActivationModule):
        m.frozen.fill_(False)
    if isinstance(m, RestrictedLinear):
        m.frozen.fill_(False)


def reset_frozen_gradients(m):
    """Set all gradients of frozen parameters to 0 (these parameters won't be updated by the optimizer)."""
    if isinstance(m, MultiActivationModule):
        m.weight.grad[:, m.frozen] = 0.0

    if isinstance(m, RestrictedLinear):
        m.weight.grad[m.frozen] = 0.0


def freeze_some_act_funcs(m, ratio=0.2):
    if isinstance(m, MultiActivationModule):
        indices = torch.max(m.weight, 0).indices

        to_freeze = (torch.rand(*m.frozen.size()) <= ratio) & ~m.frozen

        if torch.any(to_freeze):
            indices = torch.max(m.weight[:, to_freeze], 0).indices
            m.weight.data[:, to_freeze] = torch.nn.functional.one_hot(indices, m.n_funcs).T.float()
            m.frozen[to_freeze] = True


def freeze_some_weights(m, ratio=0.2, zero_ratio=0.4):
    if isinstance(m, RestrictedLinear):
        # mask-0 operation
        if torch.any(~m.frozen):
            alpha_zero = np.percentile(m.weight.data[~m.frozen].abs(), 100 * ratio * zero_ratio)
            mask_zero = (m.weight.data.abs() <= alpha_zero) & ~m.frozen

            # mask-1 operation
            alpha_one = np.percentile(-m.weight.data[~m.frozen].abs(), 100 * ratio * (1-zero_ratio))
            mask_one = (-m.weight.data.abs() <= alpha_one) & ~m.frozen

            m.frozen[mask_zero] = True
            m.frozen[mask_one] = True

            m.weight.data[mask_zero] = 0.0
            m.weight.data[mask_one] = m.weight.data[mask_one].sign()


class Model(torch.nn.Module):
    def __init__(self, shared_weight, *layer_sizes):
        super().__init__()

        layers = list()

        n_in = layer_sizes[0]

        for n_out in layer_sizes[1:]:
            layers.append(ConcatLayer(n_in, n_out, shared_weight))
            n_in += n_out

        self.network = torch.nn.Sequential(*layers)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        net_out = self.network(x)
        net_out = net_out[..., -3:]
        return self.softmax(net_out)

    def init_weights(self):
        self.apply(weight_init)

    def reset_frozen_grads(self):
        self.apply(reset_frozen_gradients)

    def freeze_act_funcs(self, ratio=0.2):
        self.apply(partial(freeze_some_act_funcs, ratio=ratio))

    def freeze_weights(self, ratio=0.2, zero_ratio=0.4):
        self.apply(partial(freeze_some_weights, ratio=ratio, zero_ratio=zero_ratio))

    def unfreeze(self):
        self.apply(unfreeze_parameters)
