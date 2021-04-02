import numpy as np
import pandas as pd
from time import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import argparse

from utils import load_data, get_edges_lists

ALL_EDGES_FILES = ['year.npy', 'share_journal.npy', 'metrics.npy', 'author_metrics.npy']
ALL_NODES_FILES = ['tfidf.npy', 'n2v.npy', 'w2v.npy']

def abs_aggregator(x1, x2):
    return np.abs(x1 - x2)

def concat_aggregator(x1, x2):
    return np.concatenate((x1,x2), axis=1)

def load_embeddings(emb_edges_files = ALL_EDGES_FILES, emb_nodes_files = ALL_NODES_FILES):
    _, node_information, node_to_index, edge_df, test_edge_df = load_data()

    y = edge_df.label.values
    y_test = test_edge_df.label.values

    if emb_edges_files:
        X = np.concatenate([np.load(f'emb_edges_train/{file}') for file in emb_edges_files], axis=1)
        X_test = np.concatenate([np.load(f'emb_edges_test/{file}') for file in emb_edges_files], axis=1)
        edges_dim = X.shape[1]

        if not emb_nodes_files:
            return X, X_test, y, y_test, edges_dim, 0

    X_nodes = np.concatenate([np.load(f'emb_nodes/{file}') for file in emb_nodes_files], axis=1)
    nodes_dim = X_nodes.shape[1]

    U, V = get_edges_lists(edge_df, node_information)
    X2 = concat_aggregator(X_nodes[U], X_nodes[V])

    U, V = get_edges_lists(test_edge_df, node_information)
    X2_test = concat_aggregator(X_nodes[U], X_nodes[V])

    if not emb_edges_files:
        return X2, X2_test, y, y_test, 0, nodes_dim

    X = np.concatenate((X, X2), axis=1)
    X_test = np.concatenate((X_test, X2_test), axis=1)
 
    return X, X_test, y, y_test, edges_dim, nodes_dim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--emb_edges_files', nargs='*', default=ALL_EDGES_FILES)
    parser.add_argument('-n', '--emb_nodes_files', nargs='*', default=ALL_NODES_FILES)
    parser.add_argument('-c', '--classifier', type=str, default='mlp')
    args = parser.parse_args()

    X, X_test, y, y_test, _, _ = load_embeddings(emb_edges_files=args.emb_edges_files,
                                           emb_nodes_files=args.emb_nodes_files)
    print('Embeddings loaded. Shape:', X.shape, '(train),', X_test.shape, '(test)') 

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y)

    print('Starting training...')

    if args.classifier == 'mlp':
        clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,32), learning_rate_init=5e-5, verbose=True, tol=3e-3))
    elif args.classifier == 'svc':
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', verbose=True))
    else:
        raise ValueError(f"Invalid classifier name. Found {args.classifier}.")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print(f"Training F1: {f1_score(y_train, clf.predict(X_train))}")
    print(f"Training Accuracy: {accuracy_score(y_train, clf.predict(X_train))}")

    print(f"Validation F1: {f1_score(y_val, y_pred)}")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)}")

    file_path = os.path.join('submissions', f"{time()}.csv")
    ids, categories = list(range(len(X_test))), clf.predict(X_test).tolist()
    pd.DataFrame(data={'id': ids, 'category': categories}).to_csv(file_path, index=False)
    print(f"\nSubmission saved at {file_path}")

