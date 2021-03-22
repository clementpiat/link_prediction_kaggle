"""
Validation F1: 0.945
"""
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier


def common_neighbors(graph, n1, n2):
    return len(list(nx.common_neighbors(graph, n1, n2)))

def jaccard_coefficient(graph, n1, n2):
    inter = common_neighbors(graph, n1, n2)
    union = len(set(graph[n1]) | set(graph[n2]))
    return inter/union

def adamic_adar(graph, n1, n2):
    inter_list = nx.common_neighbors(graph, n1, n2)
    return sum([1/np.log(graph.degree(node)) for node in inter_list])

def preferential_attachment(graph, n1, n2):
    return graph.degree(n1) * graph.degree(n2)

def shortest_path(graph, n1, n2):
    return - len(nx.shortest_path(G, source=n1, target=n2))

def katzb(graph, n1, n2):
    pass

def rooted_page_rank(graph, n1, n2):
    pass

def all_metrics(graph, n1, n2):
    return [
        common_neighbors(graph, n1, n2),
        jaccard_coefficient(graph, n1, n2),
        adamic_adar(graph, n1, n2),
        preferential_attachment(graph, n1, n2),
        shortest_path(graph, n1, n2)
    ]


train_df = pd.read_csv("training_set.txt", sep=" ", header=None)
edges = np.array(train_df)
G = nx.Graph()
for row in tqdm(edges):
    G.add_edge(row[0], row[1])

def get_X(arr):
    return np.array([all_metrics(G, row[0], row[1]) for row in tqdm(arr)])

if os.path.exists('train.npy'):
    X_train, X_val, y_train, y_val = np.load('train.npy', allow_pickle=True)
else:
    train, val = train_test_split(edges, train_size=0.7, stratify=edges[:,2])
    X_train, X_val = get_X(train), get_X(val)
    y_train, y_val = train[:,2], val[:, 2]

    with open('train.npy', 'wb') as f:
        np.save(f, [X_train, X_val, y_train, y_val])


#clf = LogisticRegression(C=1, max_iter=500)
clf = MLPClassifier(hidden_layer_sizes=(64,64), alpha=1e-4)
clf.fit(X_train, y_train)
prediction = clf.predict(X_val)

print(f"Training F1: {f1_score(y_train, clf.predict(X_train))}")
print(f"Training Accuracy: {accuracy_score(y_train, clf.predict(X_train))}")

print(f"Validation F1: {f1_score(y_val, prediction)}")
print(f"Validation Accuracy: {accuracy_score(y_val, prediction)}")