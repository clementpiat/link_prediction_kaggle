import pandas as pd 
import numpy as np
import networkx as nx 
import random as rd 
from tqdm import tqdm
from gensim.models import Word2Vec
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier

rd.seed(0)

def load_data(ratio_pairs_vw = 0.8):
    """
    Load training_set.txt and split it into 2 objects:
        - G_vw: a networkx Graph to learn the nodes representation
        - classif_edges: a list of edges to train a link prediction classifier as a numpy array

    ratio_pairs_vw: ratio of the positive pairs that will be used to learn the nodes representations
    """
    train_df = pd.read_csv("training_set.txt", sep=" ", header=None, names=["source", "target", "label"])
    nodes = set(list(train_df.source)+list(train_df.target))
    print(f"Number of nodes: {len(nodes)}")

    G_vw = nx.Graph()
    for node in nodes:
        #G_vw.add_node(node)
        G_vw.add_edge(node, node) # Small trick

    print("\nBuilding graph...")
    classif_edges = []
    unbalanced_count = 0 # A positive integer that measures the following difference: <number_of_positive_pairs> - <number_of_negative_pairs>
    for row in tqdm(np.array(train_df)):
        if row[2] and rd.random() < ratio_pairs_vw:
            G_vw.add_edge(row[0], row[1])
            G_vw.add_edge(row[1], row[0])
        elif unbalanced_count or row[2]:
            classif_edges.append(row)
            unbalanced_count += (row[2]*2 - 1) 

    return G_vw, np.array(classif_edges)


def perform_random_walks(graph, N, L):
    '''
    :param graph: networkx graph
    :param N: the number of walks for each node
    :param L: the walk length
    :return walks: the list of walks
    '''
    walks = []
    for node in tqdm(graph.nodes):
        for _ in range(N):
            walk = [str(node)]
            for _ in range(L):
                node = np.random.choice(list(graph.neighbors(node)))
                walk.append(str(node))
            walks.append(walk)

    rd.shuffle(walks)        
    return walks

def get_word_vectors():
    if os.path.exists(output_filename):
        return Word2Vec.load(output_filename).wv
    else:
        print("\nComputing word vectors...")
        # Perform random walks - call function
        walks = perform_random_walks(G_vw, num_of_walks, walk_length)
        # Learn representations of nodes - use Word2Vec
        model = Word2Vec(walks, size=embedding_size, window=window_size, workers=4, min_count=1)
        # Save the embedding vectors
        assert len(model.wv.vocab) == 27770
        model.save(output_filename)
        return model.wv

train_size = 0.7
num_of_walks=20
walk_length=10
embedding_size=512
window_size=5
output_filename="./word2vec.model"


# Load data
G_vw, classif_edges = load_data()
wv = get_word_vectors()

# Train a model
def get_X(arr):
    return np.array([wv[str(row[0])] + wv[str(row[1])] for row in arr])

train, val = train_test_split(classif_edges, train_size=train_size, stratify=classif_edges[:,2])
X_train, X_val = get_X(train), get_X(val)

#clf = LogisticRegression(C=1, max_iter=500)
clf = MLPClassifier(hidden_layer_sizes=(64,32), alpha=1e-2)
clf.fit(X_train, train[:,2])
prediction = clf.predict(X_val)

print(f"Training F1: {f1_score(train[:,2], clf.predict(X_train))}")
print(f"Training Accuracy: {accuracy_score(train[:,2], clf.predict(X_train))}")

print(f"Validation F1: {f1_score(val[:,2], prediction)}")
print(f"Validation Accuracy: {accuracy_score(val[:,2], prediction)}")