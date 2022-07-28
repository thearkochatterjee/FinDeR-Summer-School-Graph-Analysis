import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy


def GenerateGraphs(num_graphs: int, num_nodes: int):
    # Here are the categories of graph generators
    types_of_graphs = [
    lambda n: nx.complete_graph(n),
    lambda n: nx.turan_graph(n,3),
    lambda n: nx.newman_watts_strogatz_graph(n,3,0.2),
    lambda n: nx.ladder_graph(n),
    lambda n: nx.barabasi_albert_graph(n,3)
    ]
        
    # We will sample them with different weights.
    # Clusters will have uneven size
    weights = np.random.rand(len(types_of_graphs));
    weights /= sum(weights)

    # Now generate the graphs
    graphs= []
    true_labels = []
    for graph_index in np.random.choice(range(len(types_of_graphs)), num_graphs, list(weights)):
        graphs.append(types_of_graphs[graph_index](num_nodes))
        true_labels.append(graph_index)
    return graphs, true_labels

def convert2Matrix(graphs):
    matrices = []
    for g in graphs:
        matrices.append(nx.adjacency_matrix(g).todense())
    return matrices

