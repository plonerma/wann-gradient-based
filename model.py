import torch
from contextlib import contextmanager
from functools import reduce


class Discretizable:
    _is_discrete = False

    @property
    def is_discrete(self):
        return self._is_discrete

    def discretize_weight(self, alpha=1):
        if not self.is_discrete:
            self.stored_weight.data.copy_(self.weight.data)

            self.weight.data.copy_(
                (1-alpha) * self.weight
                + alpha * self.effective_weight)
            self._is_discrete = True

    def restore_weight(self):
        if self.is_discrete:
            self.weight.data.copy_(self.stored_weight.data)
            self._is_discrete = False

    def clip_weight(self):
        torch.nn.functional.hardtanh(self.weight.data, inplace=True)


class ActModule(torch.nn.Module, Discretizable):
    """Applies multiple elementwise activation functions to a tensor."""

    available_act_functions = [
        ('relu', torch.relu),
        ('sigmoid', torch.sigmoid),
        ('tanh', torch.tanh),
        #('gaussian (standard)', lambda x: torch.exp(-torch.square(x) / 2.0)),
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

    def init_weight(self, t=None):
        if t is None:
            t = self.weight
        torch.nn.init.uniform_(t.data, 0, 1)
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

    def append_nodes_out(self, n=1):
        """Adds `n` new blank output node (initialize with near zero weight).

        Will add the nodes at the end of nodes of this layer."""

        new_nodes = torch.Tensor(self.n_funcs, n)

        self.init_weight(new_nodes)

        new_weight = torch.cat([self.weight.data, new_nodes], dim=1)

        self.weight = torch.nn.Parameter(new_weight)
        self.stored_weight = torch.empty_like(self.weight)
        self.out_features = self.out_features + n

    def delete_nodes_out(self, i):
        if not isinstance(i, torch.Tensor):
            i = torch.LongTensor([i])
        keep = torch.ones(self.out_features, dtype=bool)
        keep[i] = False
        self.out_features -= i.shape[0]
        new_weight = self.weight.data[:, keep]

        self.weight = torch.nn.Parameter(new_weight)
        self.stored_weight = torch.empty_like(self.weight)


class TertiaryLinear(torch.nn.Linear, Discretizable):
    """Similar to torch.nn.Linear, with tertiary weights ($\in \{-1,0,1\}$)."""

    lambd = 0.4

    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out, bias=False)

        self.stored_weight = torch.empty_like(self.weight)

    @property
    def effective_weight(self):
        return torch.sign(torch.nn.functional.hardshrink(self.weight, lambd=self.lambd))

    def init_weight(self, t=None, near_zero=True):
        if t is None:
            t = self.weight
        if near_zero:
            torch.nn.init.uniform_(t.data, -.1, .1)
        else:
            torch.nn.init.normal_(t.data, std=.1)

    def append_nodes_out(self, n):
        """Adds `n` blank output nodes (initialize with near zero weight).

        Will add the nodes at the end of nodes of this layer.
        """
        new_nodes = torch.Tensor(n, self.in_features)

        self.init_weight(new_nodes, near_zero=True)

        new_weight = torch.cat([self.weight.data, new_nodes], dim=0)

        self.weight = torch.nn.Parameter(new_weight)
        self.stored_weight = torch.empty_like(self.weight)
        self.out_features = self.out_features + n

    def append_nodes_in(self, i, n):
        """Adds n new inputse (initialize with near zero weight).

        Rows are added starting at index i.
        """
        assert i <= self.in_features, f"Node {i} can not be input for this layer (layer has {self.in_features} inputs)."

        self.in_features = self.in_features + n
        new_weight = torch.Tensor(self.out_features, self.in_features)
        new_nodes = torch.Tensor(self.out_features, n)
        self.init_weight(new_nodes, near_zero=True)

        before = torch.arange(0, i)
        new_indices = torch.arange(i, i+n)
        after = torch.arange(i+n, self.in_features)

        old_indices = torch.cat([before, after])

        # copy old weight
        new_weight.index_copy_(
            dim=1, index=old_indices, source=self.weight.data)

        # copy new node
        new_weight.index_copy_(
            dim=1, index=new_indices, source=new_nodes)

        self.weight = torch.nn.Parameter(new_weight)
        self.stored_weight = torch.empty_like(self.weight)

    def delete_nodes_out(self, i):
        if not isinstance(i, torch.Tensor):
            i = torch.LongTensor([i])
        keep = torch.ones(self.out_features, dtype=bool)
        keep[i] = False
        self.out_features -= i.shape[0]
        self.weight = torch.nn.Parameter(self.weight.data[keep, :])
        self.stored_weight = torch.empty_like(self.weight)

    def delete_nodes_in(self, i):
        if not isinstance(i, torch.Tensor):
            i = torch.LongTensor([i])
        keep = torch.ones(self.in_features, dtype=bool)
        keep[i] = False
        self.in_features -= i.shape[0]
        self.weight = torch.nn.Parameter(self.weight.data[:, keep])
        self.stored_weight = torch.empty_like(self.weight)


class ConcatLayer(torch.nn.Module):
    """Contatenates output of the active nodes and prior nodes."""

    gamma = 0.3

    def __init__(self, n_in, n_out, shared_weight):
        super().__init__()

        self.linear = TertiaryLinear(n_in, n_out)
        self.activation = ActModule(n_out)

        self.shared_weight = shared_weight

    def init_weight(self):
        self.linear.init_weight()
        self.activation.init_weight()

    def discretize_weight(self, alpha=1):
        self.linear.discretize_weight(alpha=alpha)
        self.activation.discretize_weight(alpha=alpha)

    def restore_weight(self):
        self.linear.restore_weight()
        self.activation.restore_weight()

    def clip_weight(self):
        self.linear.clip_weight()
        self.activation.clip_weight()

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def forward(self, x):
        linear = self.linear(x) * self.shared_weight[:, None, None]

        inner_out = self.activation(linear)

        return torch.cat([x, inner_out], dim=-1)

    def append_nodes_in(self, i, n=1):
        self.linear.append_nodes_in(i, n)

    def append_nodes_out(self, n=1):
        i = self.in_features + self.out_features
        self.linear.append_nodes_out(n)
        self.activation.append_nodes_out(n)
        return i

    def delete_nodes_in(self, i):
        self.linear.delete_nodes_in(i)

    def delete_nodes_out(self, i):
        self.linear.delete_nodes_out(i)
        self.activation.delete_nodes_out(i)

    def nodes_without_input(self):
        # inputs are spread across dimension 1
        out_indices = torch.arange(self.out_features)
        return out_indices[(self.linear.weight.data.abs() < self.gamma).all(1)]


class Model(torch.nn.Module):
    blank_nodes = 6

    def __init__(self, shared_weight, n_in, n_out, hidden_layer_sizes=None):
        super().__init__()

        self.in_features = n_in
        self.out_features = n_out
        self.shared_weight = shared_weight

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [self.blank_nodes]

        self.hidden_layers = torch.nn.ModuleList()

        for n_o in hidden_layer_sizes:
            n_o = int(n_o)
            self.hidden_layers.append(
                ConcatLayer(n_in, n_o, shared_weight))
            n_in += n_o

        self.output_layer = ConcatLayer(n_in, n_out, shared_weight)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        for layer in self.layers():
            x = layer(x)

        net_out = x[..., -self.out_features:]

        return self.softmax(net_out)

    def numel(self):
        return sum([p.numel() for p in self.parameters()])

    def nodes_without_input(self, hidden_only=False):
        return sum([layer.nodes_without_input().shape[0] for layer in self.layers(hidden_only)])

    def layer_sizes(self, hidden_only=False):
        return torch.Tensor([layer.out_features for layer in self.layers(hidden_only)])

    def layers(self, hidden_only=False):
        yield from self.hidden_layers
        if not hidden_only:
            yield self.output_layer

    def grow(self):
        grew = False

        for layer_i, layer in enumerate(self.hidden_layers):
            n = max(self.blank_nodes - layer.nodes_without_input().shape[0], 0)

            if n > 0:
                grew = True

                # add new node to layer
                node_i = layer.append_nodes_out(n)

                # update later layers (expand weight matrices accordingly)
                for later_hidden in self.hidden_layers[layer_i+1:]:
                    later_hidden.append_nodes_in(node_i, n)
                self.output_layer.append_nodes_in(node_i, n)

        last_hidden = self.hidden_layers[-1]

        if last_hidden.out_features > self.blank_nodes:
            grew = True

            # new blank layer needed
            n_in = self.output_layer.in_features

            layer = ConcatLayer(n_in, self.blank_nodes, self.shared_weight)
            layer.init_weight()

            self.hidden_layers.append(layer)

            self.output_layer.append_nodes_in(n_in, n=self.blank_nodes)

        return grew

    def cleanup(self):
        for i, layer in enumerate(self.hidden_layers):
            nodes_to_delete = layer.nodes_without_input()
            if nodes_to_delete.shape[0] > 0:
                layer.delete_nodes_out(nodes_to_delete)

                # add offset for global indices
                nodes_to_delete += layer.in_features
                for later_hidden in self.hidden_layers[i+1:]:
                    later_hidden.delete_nodes_in(nodes_to_delete)
                self.output_layer.delete_nodes_in(nodes_to_delete)

        # go though layers in reverse order to preserve correctness of indices
        for i in reversed(range(len(self.hidden_layers))):
            if self.hidden_layers[i].out_features == 0:
                del self.hidden_layers[i]

    def to_dict(self):
        return {
            'state': self.state_dict(),
            'hidden_layer_sizes': self.layer_sizes(hidden_only=True),
        }

    @classmethod
    def from_dict(cls, shared_weight, n_in, n_out, d):
        m = cls(shared_weight, n_in, n_out, hidden_layer_sizes=d['hidden_layer_sizes'])
        m.load_state_dict(d['state'])
        return m

    @contextmanager
    def discrete(self, alpha=1):
        try:
            self.discretize(alpha=alpha)
            yield
        finally:
            self.restore()

    def discretize(self, alpha=1):
        for layer in self.layers():
            layer.discretize_weight(alpha=alpha)

    def restore(self):
        for layer in self.layers():
            layer.restore_weight()

    def clip(self):
        for layer in self.layers():
            layer.clip_weight()

    def init_weight(self):
        for layer in self.layers():
            layer.init_weight()


def write_hist(writer, model, epoch):
    effective_weights = list()
    actual_weights = list()

    for layer in model.hidden_layers:
        m = layer.linear
        effective_weights.append(m.effective_weight.reshape(-1))
        actual_weights.append(m.weight.reshape(-1))

    effective_weights = torch.cat(effective_weights)
    actual_weights = torch.cat(actual_weights)

    writer.add_histogram("effective weights", effective_weights, epoch)
    writer.add_histogram("actual weights", actual_weights, epoch)
