import torch
from torch.autograd import Variable

from model import Model

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n_in = 16*16
n_out = 10

w = Variable(torch.Tensor())

d = torch.load('tmp_model_16.pt')
model = Model.from_dict(w, n_in+1, n_out, d)
model.cleanup()



square = torch.linspace(-1, 1, 16).unsqueeze(dim=0).repeat(16, 1)

x = square.reshape(-1) * 0.5
y = square.T.reshape(-1) - x*0.1

pos_in = torch.cat([y, torch.Tensor([-1.2])], dim=0)
pos_in_x = torch.cat([x, torch.Tensor([0])], dim=0)




pos_out = torch.linspace(-1, 1, n_out)

pos_hidden = [torch.nn.Parameter(torch.zeros(int(size))) for size in model.layer_sizes(hidden_only=True)]

for p in pos_hidden:
    torch.nn.init.normal_(p.data, std=.5)

optimizer = torch.optim.SGD(pos_hidden, lr=0.2)

for epoch in range(20):
    optimizer.zero_grad()

    loss = torch.tensor(0.)

    nodes_in = pos_in
    for nodes_out, layer in zip(pos_hidden + [pos_out], model.layers()):

        dist = (nodes_in.unsqueeze(dim=0) - nodes_out.unsqueeze(dim=1)).abs()
        loss += (dist * layer.linear.effective_weight.abs()).sum()

        inner_dist = (nodes_out.unsqueeze(dim=0) - nodes_out.unsqueeze(dim=1)).abs()

        loss += (1 / (inner_dist + 1)**2).sum()

        loss += (nodes_out * nodes_out).sum()

        nodes_in = torch.cat([nodes_in, nodes_out])

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        for p in pos_hidden:
            p.data = torch.nn.functional.normalize(p, dim=0)

y = torch.cat([pos_in] + pos_hidden + [pos_out], dim=0)
x = torch.zeros_like(y)

layer_x = torch.linspace(-1, 1, 1 + len(model.hidden_layers))

x[:n_in+1] = pos_in_x - 1.8

for i, layer in enumerate(model.layers()):
    x[layer.in_features:layer.in_features + layer.out_features] = layer_x[i]

node_names = (
    # inputs
    [f"$x_{{{i}}}$" for i in range(n_in)]

    # bias
    + ["$b$"]

    # hidden nodes
    + [
        f"$h_{{{layer_i}, {node_i}}}$"
        for layer_i, layer_size in enumerate(model.layer_sizes())
        for node_i in range(int(layer_size))
    ]

    # outputs
    + [f"$y_{{{i}}}$" for i in range(n_out)]
)

# Colors
color = (
    ['#fdb462'] * n_in         # inputs
    + ['#ffed6f']                   # bias
    + ['#80b1d3'] * int(model.layer_sizes().sum())   # hidden
    + ['#b3de69'] * n_out      # outputs
)


g = nx.DiGraph()
g.add_nodes_from(node_names)


pos = dict(zip(node_names, np.array([x.detach().numpy(), y.detach().numpy()]).T))

ax = plt.gca()
ax.set_axis_off()

nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_color=color, node_size=100)

ns = node_names[-n_out:]

nx.draw_networkx_labels(
    g, ax=ax, pos=pos, labels=dict(zip(ns, ns)),
    font_size=8,
)

edge_params = dict(
        edge_cmap=plt.get_cmap('tab10'),
        alpha=.6, ax=ax, pos=pos,
        edge_vmin=0, edge_vmax=9,
        arrows=True,
    )

edge_col = list()
edgelist = list()

for layer in model.layers():
    connections = layer.linear.effective_weight.abs() >= layer.linear.lambd

    for dst, src in zip(*np.where(connections.detach().numpy())):
        edgelist.append((node_names[src], node_names[dst + layer.in_features]))
        edge_col.append(
            2 if layer.linear.weight[dst][src] > 0
            else 3
        )

nx.draw_networkx_edges(g, edgelist=edgelist, edge_color=edge_col,
                       width=1, arrowstyle='-',
                       min_source_margin=10, min_target_margin=5,
                       **edge_params)


plt.show()
