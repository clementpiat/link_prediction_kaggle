import numpy as np
from node2vec import Node2Vec
import argparse

from utils import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimensions", type=int, default=128)
    parser.add_argument("-wl", "--walk_length", type=int, default=8)
    parser.add_argument("-n", "--num_walks", type=int, default=50)
    parser.add_argument("-win", "--window", type=int, default=6)
    parser.add_argument("-bw", "--batch_words", type=int, default=4)
    parser.add_argument("-mc", "--min_count", type=int, default=1)
    args = parser.parse_args()
    
    graph, *_ = load_data()

    node2vec = Node2Vec(graph, dimensions=args.dimensions, walk_length=args.walk_length, num_walks=args.num_walks, p=1, q=1)
    model = node2vec.fit(window=args.window, min_count=args.min_count, batch_words=args.batch_words)

    np.save('emb_nodes/n2v.npy', [model.wv[str(node)] for node in graph.nodes()])
    