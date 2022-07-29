from cProfile import label
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy


def GenerateGraphs(num_graphs: int, num_nodes: int):
    """
    Generating the Graph Data for a randomized category.

    @params:
    num_graph: number of graphs to generate
    num_nodes: number of nodes per graph
    """
    # Here are the categories of graph generators
    types_of_graphs = [
    lambda n: nx.complete_graph(n),
    lambda n: nx.turan_graph(n,3),
    lambda n: nx.newman_watts_strogatz_graph(n,3,0.2),
    lambda n: nx.ladder_graph(int(n/2)),
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
    """
    Converts list of graphs to list of Adjacency Matrices

    @params:
    graphs: list of graphs
    """
    matrices = []
    for g in graphs:
        matrices.append(nx.adjacency_matrix(g).todense())
    return matrices

def convert2Graph(matricies):
    """
    Converts list of Adjacency Matrices to list of graphs

    @params:
    matricies: list of Adjacency Matrices
    """
    g = []
    for m in matricies:
        g.append(nx.from_numpy_matrix(m))
    return g

def plotGraphs(graphs, labels=None, num_plots: int = -1, show=True):
    """
    Plots graphs.

    @params
    graphs: list of graphs
    labels: list of labels for the cateogory type of the graph

    num_plots: number of graphs to plot. (default of -1 will plot all graphs)
    """
    num = num_plots
    if num_plots == -1:
        num = len(graphs)
    for i, g in enumerate(graphs[:num]):
        plt.figure()
        if labels is not None:
            plt.title(f'Category {labels[i]}')
        nx.draw(g, with_labels=True, font_weight='bold')
    if show:
        plt.show()

def make_flat(matrices):
    vals = []
    for m in matrices:
        vals.append(m.flatten())
    return vals

def toarray(matrices):
    vals = []
    for m in matrices:
        vals.append(np.asarray(m))
    return vals

# graphs, lab = GenerateGraphs(1, 10)
# plotGraphs(graphs)
# matrices = convert2Matrix(graphs)
# print(type(matrices[0].to_array()))
# max = graphs[0].number_of_nodes()
# min = graphs[0].number_of_nodes()
# for g in graphs:
#     if g.number_of_nodes() > max:
#         max = g.number_of_nodes()
#     elif g.number_of_nodes() < min:
#         min = g.number_of_nodes()
# print(max)
# print(min)
