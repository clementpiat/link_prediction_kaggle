import numpy as np

from utils import load_data, get_edges_lists

if __name__ == '__main__':
    _, node_information, node_to_index, edge_df, test_edge_df = load_data()

    for name, df in [('train', edge_df), ('test', test_edge_df)]:
        U, V = get_edges_lists(df, node_information)

        share_journal = (node_information.journal.values[U] == node_information.journal.values[V]).astype(int)[:,None]
        with open(f'emb_edges_{name}/share_journal.npy', 'wb') as f:
            np.save(f, share_journal)

        year = - np.abs(node_information.year.values[U] - node_information.year.values[V])[:,None]
        with open(f'emb_edges_{name}/year.npy', 'wb') as f:
            np.save(f, year)


