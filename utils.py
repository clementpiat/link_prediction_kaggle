import networkx as nx
import pandas as pd
import numpy as np
import random as rd

def load_node_information():
    return pd.read_csv('node_information.csv', header=None, names=['node', 'year', 'title', 'authors', 'journal', 'abstract']).fillna('')

def get_edges_lists(edge_df, node_information):
    node_to_index = {node: i for i, node in enumerate(node_information.node.values)}
    U = np.array([node_to_index[node] for node in edge_df.source.values])
    V = np.array([node_to_index[node] for node in edge_df.target.values])
    return U, V

def load_data(cut=0):
    """
    Load training_set.txt and split it into 2 objects:
        - G_vw: a networkx Graph to learn the nodes representation
        - classif_edges: a list of edges to train a link prediction classifier as a numpy array

    ratio_pairs_vw: ratio of the positive pairs that will be used to learn the nodes representations
    """
    rd.seed(0)
    node_information = load_node_information()
    edge_df = pd.read_csv("training_set.txt", sep=" ", header=None, names=["source", "target", "label"])
    test_edge_df = pd.read_csv("testing_set.txt", sep=" ", header=None, names=["source", "target", "label"])
    validation_split = np.load("validation_split.npy")

    graph = nx.Graph()

    for node in node_information.node:
        graph.add_edge(node, node)

    train_edge_df = edge_df[validation_split]
    for source, target, label in zip(train_edge_df.source.values, train_edge_df.target.values, train_edge_df.label.values):
        if label and rd.random() >= cut:
            graph.add_edge(source, target)

    node_to_index = {node: i for i, node in enumerate(node_information.node.values)}

    print(f"Loaded graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"Number of connected components: {nx.number_connected_components(graph)}")

    return graph, node_information, node_to_index, edge_df, test_edge_df