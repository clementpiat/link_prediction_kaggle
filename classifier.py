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

from utils import load_data, get_edges_lists

def abs_aggregator(x1, x2):
    return np.abs(x1, x2)

if __name__ == '__main__':
    graph, node_information, node_to_index, edge_df, test_edge_df = load_data()

    print('Concatenating all embeddings...')

    emb_edges_files = ['year.npy', 'share_journal.npy', 'metrics.npy']
    X = np.concatenate([np.load(f'emb_edges_train/{file}') for file in emb_edges_files], axis=1)
    X_test = np.concatenate([np.load(f'emb_edges_test/{file}') for file in emb_edges_files], axis=1)
    
    emb_nodes_files = ['tfidf.npy']
    X_nodes = np.concatenate([np.load(f'emb_nodes/{file}') for file in emb_nodes_files], axis=1)

    U, V = get_edges_lists(edge_df, node_information)
    X = np.concatenate((X, abs_aggregator(X_nodes[U], X_nodes[V])), axis=1)

    U, V = get_edges_lists(test_edge_df, node_information)
    X_test = np.concatenate((X_test, abs_aggregator(X_nodes[U], X_nodes[V])), axis=1)

    print('Embeddings loaded. Shape:', X.shape, '(train),', X_test.shape, '(test)')

    y = edge_df.label.values
    y_test = test_edge_df.label.values
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y)

    print('Starting training...')

    #clf = MLPClassifier(hidden_layer_sizes=(64,32), learning_rate_init=1e-5, alpha=1e-2, verbose=True)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', verbose=True))

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

