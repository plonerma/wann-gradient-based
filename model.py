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

    def add_node_out(self):
        """Adds a new blank output node (initialize with near zero weight).

        Will add the node at the end of nodes of this layer."""

        new_node = torch.Tensor(self.n_funcs, 1)

        self.init_weight(new_node)

        new_weight = torch.cat([self.weight.data, new_node], dim=1)

        self.weight = torch.nn.Parameter(new_weight)
        self.stored_weight = torch.empty_like(self.weight)
        self.out_features = self.out_features + 1


class TertiaryLinear(torch.nn.Linear, Discretizable):
    """Similar to torch.nn.Linear, with tertiary weights ($\in \{-1,0,1\}$)."""

    lambd = 0.4

    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out, bias=False)

        self.stored_weight = torch.empty_like(self.weight)

    @property
    def effective_weight(self):
        return torch.sign(torch.nn.functional.hardshrink(self.weight, lambd=self.lambd))

    def init_weight(self, t=None, near_zero=False):
        if t is None:
            t = self.weight
        if near_zero:
            torch.nn.init.uniform_(t.data, -.1, .1)
        else:
            torch.nn.init.normal_(t.data, std=.1)

    def add_node_out(self):
        """Adds a new blank output node (initialize with near zero weight).

        Will add the node at the end of nodes of this layer.
        """
        new_node = torch.Tensor(1, self.in_features)

        self.init_weight(new_node, near_zero=True)

        new_weight = torch.cat([self.weight.data, new_node], dim=0)

        self.weight = torch.nn.Parameter(new_weight)
        self.stored_weight = torch.empty_like(self.weight)
        self.out_features = self.out_features + 1

    def add_node_in(self, i):
        """Adds a new blank input node (initialize with near zero weight).

        Row is added at index i.
        """
        assert i <= self.in_features

        self.in_features = self.in_features + 1
        new_weight = torch.Tensor(self.out_features, self.in_features)
        new_node = torch.Tensor(self.out_features, 1)
        self.init_weight(new_node)

        before = torch.arange(0, i)
        after = torch.arange(i+1, self.in_features)

        # copy old weight
        new_weight.index_copy_(
            dim=1, index=torch.cat([before, after]), source=self.weight.data)

        # copy new node
        new_weight.index_copy_(
            dim=1, index=torch.LongTensor([i]), source=new_node)

        self.weight = torch.nn.Parameter(new_weight)
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

    def add_node_in(self, i):
        print(f"adding node #{i} to inputs")
        self.linear.add_node_in(i)

    def add_node_out(self):
        self.linear.add_node_out()
        self.activation.add_node_out()
        i = self.in_features + self.out_features - 1
        return i

    def nodes_without_input(self):
        inactive_conns = self.linear.weight.data.abs() < self.gamma
        # inputs are spread across dimension 1
        return inactive_conns.all(1).sum()

    def parameters(self):
        yield self.linear.weight
        yield self.activation.weight


class Model:
    blank_nodes = 6

    def __init__(self, shared_weight, n_in, n_out,
                 hidden_layer_sizes=None, grow=False):
        super().__init__()

        assert (hidden_layer_sizes is not None) != grow, "Model can EITHER grow OR have static hidden layer sizes."

        self.in_features = n_in
        self.out_features = n_out
        self.shared_weight = shared_weight

        if grow:
            hidden_layer_sizes = [self.blank_nodes]

        self.hidden_layers = list()

        for n_out in hidden_layer_sizes:
            self.hidden_layers.append(
                ConcatLayer(n_in, n_out, shared_weight))
            n_in += n_out

        self.output_layer = ConcatLayer(n_in, n_out, shared_weight)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        net_out = self.output_layer(x)

        net_out = net_out[..., -self.out_features:]

        return self.softmax(net_out)

    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        for layer in self.layers():
            yield from layer.parameters()

    def numel(self):
        return sum([p.numel() for p in self.parameters()])

    def nodes_without_input(self, hidden_only=False):
        layers = self.hidden_layers if hidden_only else self.layers()
        return sum([layer.nodes_without_input() for layer in layers])

    def layer_sizes(self):
        return torch.Tensor([layer.out_features for layer in self.layers()])

    def layers(self):
        yield from self.hidden_layers
        yield self.output_layer

    def grow(self):
        grew = False

        for layer_i, layer in enumerate(self.hidden_layers):
            for _ in range(max(self.blank_nodes - layer.nodes_without_input(), 0)):
                print(f"Adding new node to layer #{layer_i}")
                grew = True

                # add new node to layer
                node_i = layer.add_node_out()

                # update later layers (expand weight matrices accordingly)

                for later_hidden in self.hidden_layers[layer_i+1:]:
                    later_hidden.add_node_in(node_i)

                self.output_layer.add_node_in(node_i)

        last_hidden = self.hidden_layers[-1]

        if last_hidden.out_features > self.blank_nodes:
            grew = True

            print("Adding new layer")

            # new blank layer needed
            n_in = last_hidden.in_features + last_hidden.out_features
            layer = GrowingConcatLayer(n_in, 1, self.shared_weight)
            layer.init_weight()
            self.hidden_layers.append(layer)

            self.output_layer.add_node_in(n_in + layer.out_features - 1)

        return grew

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

    def init_weights(self):
        for layer in self.layers():
            layer.init_weight()

    def modules(self):
        for m in self.layers():
            yield from m.modules()
            yield m
        yield self.softmax

def write_hist(writer, model, epoch):
    effective_weights = list()
    actual_weights = list()

    for layer in model.hidden_layers:
        m = layer.linear
        effective_weights.append(m.effective_weight.reshape(-1))
        actual_weights.append(m.weight.reshape(-1))

    effective_weights = torch.cat(effective_weights)
    actual_weights = torch.cat(actual_weights)

    if torch.any(torch.isnan(effective_weights)):
        print("NAN!!!")
        print(effective_weights)

    if torch.any(torch.isnan(actual_weights)):
        print("NAN!!!")
        print(effective_weights)

    writer.add_histogram("effective weights", effective_weights, epoch)
    writer.add_histogram("actual weights", actual_weights, epoch)
