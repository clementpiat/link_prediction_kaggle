import numpy as np
from gensim.models import Word2Vec
import argparse
from tqdm import tqdm
import random as rd

from utils import load_data

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embedding_size", type=int, default=256)
    parser.add_argument("-wl", "--walk_length", type=int, default=10)
    parser.add_argument("-n", "--num_walks", type=int, default=20)
    parser.add_argument("-win", "--window_size", type=int, default=5)
    args = parser.parse_args()
    
    graph, *_ = load_data()

    # Perform random walks - call function
    walks = perform_random_walks(graph, args.num_walks, args.walk_length)
    # Learn representations of nodes - use Word2Vec
    model = Word2Vec(walks, size=args.embedding_size, window=args.window_size, workers=4, min_count=1)
    # Save the embedding vectors
    assert len(model.wv.vocab) == 27770


    np.save('emb_nodes/w2v.npy', [model.wv[str(node)] for node in graph.nodes()])
    