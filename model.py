import torch
from contextlib import contextmanager
from functools import reduce


class Discretizable:
    is_discrete = False

    def alpha_blend(self, alpha=1):
        if not self.is_discrete:
            with torch.no_grad():
                self.stored_weight.data.copy_(self.weight.data)

                self.weight.data.copy_(
                    (1-alpha) * self.weight
                    + alpha * self.quantized_weight)
                self.is_discrete = True

        # TODO: In Activation Module: Test whether dividing the weight by the
        # sum makes a differences for how the gradient is calculated!

    def restore_weight(self):
        if self.is_discrete:
            with torch.no_grad():
                self.weight.data.copy_(self.stored_weight.data)
                self.is_discrete = False

    def clip_weight(self):
        with torch.no_grad():
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
        self.weight.data /= length

    @property
    def quantized_weight(self):
        indices = torch.max(self.weight, 0).indices
        return torch.nn.functional.one_hot(indices, self.n_funcs).T.float()

    def forward(self, x):
        # divided again to add division to gradient
        length = torch.norm(self.weight.data, dim=0).unsqueeze(dim=0)
        if (length == 0).any():
            length = 1

        coefficients = self.weight / length

        y = torch.zeros_like(x)

        for act_i, act_func in enumerate(self.funcs):
            y = y.add(act_func(x).mul(coefficients[act_i, :]))

        return y

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

    def __init__(self, n_in, n_out, lambd=0.4):
        super().__init__(n_in, n_out, bias=False)

        self.stored_weight = torch.empty_like(self.weight)
        self.lambd = lambd

    @property
    def quantized_weight(self):
        return torch.sign(torch.nn.functional.hardshrink(self.weight, lambd=self.lambd))

    def init_weight(self, t=None, near_zero=True):
        if t is None:
            t = self.weight
        if near_zero:
            torch.nn.init.uniform_(t.data, -.1, .1)
        else:
            torch.nn.init.xavier_normal_(t.data)

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

    def __init__(self, n_in, n_out, shared_weight):
        super().__init__()

        self.linear = TertiaryLinear(n_in, n_out)
        self.activation = ActModule(n_out)

        self.shared_weight = shared_weight

    def init_weight(self, near_zero=True):
        self.linear.init_weight(near_zero=near_zero)
        self.activation.init_weight()

    def alpha_blend(self, alpha=1):
        self.linear.alpha_blend(alpha=alpha)
        self.activation.alpha_blend(alpha=alpha)

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

    def nodes_without_input(self, gamma=0.4):
        # inputs are spread across dimension 1
        out_indices = torch.arange(self.out_features)
        return out_indices[(self.linear.weight.data.abs() < gamma).all(1)]
    
    def recieves_input_from(self, gamma=0.4):
        return (self.linear.weight.data.abs() >= gamma).any(0)


class Model(torch.nn.Module):
    blank_nodes = 6

    def __init__(self, shared_weight, n_in, n_out, hidden_layer_sizes=None, gamma=0.4):
        super().__init__()

        self.in_features = n_in
        self.out_features = n_out
        self.shared_weight = shared_weight
        self.gamma = gamma

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [self.blank_nodes]
            self.growing = True
        else:
            self.growing = False

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
        return sum([layer.nodes_without_input(self.gamma).shape[0] for layer in self.layers(hidden_only)])

    def layer_sizes(self, hidden_only=True):
        return torch.Tensor([layer.out_features for layer in self.layers(hidden_only)])

    def layers(self, hidden_only=False):
        yield from self.hidden_layers
        if not hidden_only:
            yield self.output_layer

    def grow(self):
        grew = False

        for layer_i, layer in enumerate(self.hidden_layers):
            n = max(self.blank_nodes - layer.nodes_without_input(self.gamma).shape[0], 0)

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
        deleted_something = False
    
        # delete nodes without input
        for i, layer in enumerate(self.hidden_layers):
            self.delete_nodes(layer.nodes_without_input(self.gamma) + layer.in_features)
        
        # delete nodes that aren't used as inputs in later layers
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            
            used_as_input = torch.zeros(layer.out_features, dtype=bool)
            
            for later_layer in list(self.layers(hidden_only=False))[i+1:]:
                used_as_input = used_as_input | later_layer.recieves_input_from()[layer.in_features:layer.in_features+layer.out_features]

            self.delete_nodes(torch.arange(layer.out_features)[~used_as_input] + layer.in_features)
            
        self.delete_empty_layers()

    def delete_empty_layers(self):
        # go though layers in reverse order to preserve correctness of indices
        for i in reversed(range(len(self.hidden_layers))):
            if self.hidden_layers[i].out_features == 0:
                del self.hidden_layers[i]
    
    def delete_nodes(self, nodes):
        if nodes.shape[0] == 0:
            return

        for layer in self.layers():    
            in_nodes = (nodes < layer.in_features)
            out_nodes = (nodes >= layer.in_features) & (nodes < (layer.in_features + layer.out_features))
            layer.delete_nodes_in(nodes[in_nodes])
            layer.delete_nodes_out(nodes[out_nodes] - layer.in_features)  

    def to_dict(self):
        return {
            'state': self.state_dict(),
            'hidden_layer_sizes': self.layer_sizes(hidden_only=True),
        }

    @classmethod
    def from_dict(cls, shared_weight, state):
        if 'hidden_layers.0.linear.weight' in state:
            n_in = state['hidden_layers.0.linear.weight'].size()[1]
        else:
            n_in = state['output_layer.linear.weight'].size()[1]
        
        n_out = state['output_layer.linear.weight'].size()[0]
        
        hidden_layer_sizes = list()
        for k, v in state.items():
            if k.startswith('hidden_layers.') and k.endswith('.linear.weight'):
                hidden_layer_sizes.append(v.size()[0])
                
        m = cls(shared_weight, n_in, n_out, hidden_layer_sizes=hidden_layer_sizes)
        m.load_state_dict(state)
        return m

    @contextmanager
    def discrete(self):
        with self.alpha_blend(alpha=1):
            yield

    @contextmanager
    def alpha_blend(self, alpha=1):
        try:
            for layer in self.layers():
                layer.alpha_blend(alpha=alpha)
            yield
        finally:
            for layer in self.layers():
                layer.restore_weight()

    def clip(self):
        for layer in self.layers():
            layer.clip_weight()

    def init_weight(self):
        for layer in self.hidden_layers:
            layer.init_weight()
        self.output_layer.init_weight(near_zero=self.growing)


def write_hist(writer, model, epoch):
    quantized_weights = list()
    actual_weights = list()

    for layer in model.hidden_layers:
        m = layer.linear
        quantized_weights.append(m.quantized_weight.reshape(-1))
        actual_weights.append(m.weight.reshape(-1))

    quantized_weights = torch.cat(quantized_weights)
    actual_weights = torch.cat(actual_weights)

    writer.add_histogram("quantized weights", quantized_weights, epoch)
    writer.add_histogram("actual weights", actual_weights, epoch)
