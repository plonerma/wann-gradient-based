import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def add_layer(g, current, type, layer, start_index=0):
    edges = list()
    
    
    for prev in g.nodes:       
        w = layer.linear.weight[:, g.nodes[prev]['start_index']:g.nodes[prev]['start_index'] + g.nodes[prev]['n_nodes']]
        w = float(w.abs().sum())
        
        if w > 0:
            edges.append((prev, current, dict(weight=w)))

    g.add_node(current, type=type, n_nodes=layer.linear.out_features, start_index=start_index)
    
    g.add_edges_from(edges)
    return current
    

def graph_from_model(model):
    with model.discrete():
        g = nx.DiGraph()

        g.add_node('input', type='in', n_nodes=model.in_features, start_index=0)

        start_index = model.in_features

        for i, layer in enumerate(model.hidden_layers):
            node = add_layer(g, f'hidden {i+1}', type='hidden', layer=layer, start_index=start_index)

            start_index += g.nodes[node]['n_nodes']

        add_layer(g, 'output', type='out', layer=model.output_layer, start_index=start_index)
    return g

N = 100
start=0.05

colors = np.zeros((N, 4))
colors[:, -1] = np.linspace(start,1,colors.shape[0])
cm_transparent_to_black = matplotlib.colors.ListedColormap(colors)

colors = np.empty((N, 4))
colors[:, 0] = colors[:, 1] = colors[:, 2] = np.linspace(1-start,0,colors.shape[0])
colors[:, -1] = 1
cm_white_to_black = matplotlib.colors.ListedColormap(colors)


def draw_model_graph(g):
    type_colors = {
        'in': '#fdb462',
        'hidden': '#80b1d3',
        'out': '#b3de69'
    }

    positions = np.zeros((len(g), 2))
    positions[:, 0] = np.arange(len(g))
    positions[:, 1] = 0

    positions = nx.drawing.layout.rescale_layout(positions)


    positions = {
        n: positions[i, :] for i, n in enumerate(g.nodes)
    }

    params = dict(
        pos=positions,

        node_size=[
            g.nodes[n]['n_nodes'] for n in g.nodes
        ],

        node_color=[type_colors[g.nodes[n]['type']] for n in g.nodes()],
        cmap=plt.get_cmap('tab20')
    )

    nx.draw_networkx_nodes(g, **params)

    #nx.draw_networkx_labels(g, positions, font_size=8)

    edge_weights = np.array([g.edges[e]['weight'] for e in g.edges])

    nx.draw_networkx_edges(
        g, pos=positions,
        connectionstyle='arc3, rad=-0.75',
        edge_color=edge_weights,
        edge_cmap=cm_transparent_to_black,
        edge_vmin=0,
        edge_vmax=edge_weights.max(),
        width=2.0,
    )
    plt.gca().set_ylim([-.2,1])
    
    
    mappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, edge_weights.max()), cmap=cm_white_to_black)
    plt.colorbar(mappable).outline.set_visible(False)
    
    plt.gca().set_axis_off()

    