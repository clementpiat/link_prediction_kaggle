from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from utils import load_data, get_edges_lists
from classifier import load_embeddings
from model import Net

ALL_EDGES_FILES = ['year.npy', 'share_journal.npy', 'metrics.npy']#, 'author_metrics.npy']
ALL_NODES_FILES = ['tfidf.npy', 'n2v.npy']

def train_and_validate(model, train_loader,val_loader,args):
    loss_fct = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.05)

    for e in range(args.epochs):
        losses = 0
        for data in train_loader:
            x,y = data
            outputs = model(x)
            
            optimizer.zero_grad()
            loss = loss_fct(outputs, y)
            loss.backward()
            optimizer.step()

            losses += loss.item()
        
        print(f"Epoch {e+1}: {losses}")

        labels = []
        predictions = []
        for data in val_loader:
            x,y = data
            outputs = model(x)

            labels.extend(y.detach().numpy())
            predictions.extend(outputs.detach().numpy())

        print(f"Validation F1: {f1_score(labels, np.round(predictions))}")
        print(f"Validation Accuracy: {accuracy_score(labels, np.round(predictions))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--emb_edges_files', nargs='*', default=ALL_EDGES_FILES)
    parser.add_argument('-n', '--emb_nodes_files', nargs='*', default=ALL_NODES_FILES)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-ne', '--epochs', type=int, default=5)
    args = parser.parse_args()

    X, X_test, y, y_test, edges_dim, nodes_dim = load_embeddings(emb_edges_files=args.emb_edges_files,
                                                                emb_nodes_files=args.emb_nodes_files)
    print('Embeddings loaded. Shape:', X.shape, '(train),', X_test.shape, '(test)') 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y)
    validation_split = np.load("validation_split.npy")
    X_train, y_train = X[validation_split], y[validation_split]
    X_val, y_val = X[np.logical_not(validation_split)], y[np.logical_not(validation_split)]

    X_train, X_val, y_train, y_val = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(y_train), torch.Tensor(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = Net(edges_dim, nodes_dim)
    print('Starting training...')
    train_and_validate(model, train_loader,val_loader,args)