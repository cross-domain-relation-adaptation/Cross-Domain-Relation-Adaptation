import torch
import torch_geometric
from torch_geometric.utils import to_undirected


def my_train_cross_train_cross_test_split_edges(data, test_ratio=0.1):
    def get_upper_triangular_portion(row, col):
        mask = (row < col)
        return row[mask], col[mask]

    def randomize_edges_order(row, col):
        perm = torch.randperm(row.shape[0])
        return row[perm], col[perm]

    def get_hetro_homo_split(row, col, y):
        mask_hetro = (data.y[row] != data.y[col])
        return (row[mask_hetro], col[mask_hetro]), (row[~mask_hetro], col[~mask_hetro])

    def get_pos_edges_split(row_hetro, col_hetro, row_homo, col_homo, num_test):
        n_t = num_test

        r, c = row_hetro[:n_t], col_hetro[:n_t]
        test_pos_edge_index = torch.stack([r, c], dim=0)

        r = torch.cat([row_hetro[n_t:], row_homo], dim=0)
        c = torch.cat([row_hetro[n_t:], row_homo], dim=0)
        train_pos_edge_index = torch.stack([r, c], dim=0)

        train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        return train_pos_edge_index, test_pos_edge_index

    def get_neg_edges_split(num_nodes, row, col, y, num_test):
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero().t()
        maks_hetro = (y[neg_row] != y[neg_col])
        neg_row_hetro, neg_col_hetro = neg_row[maks_hetro], neg_col[maks_hetro]
        perm = random.sample(
            range(neg_row_hetro.size(0)),
            min(num_test, neg_row_hetro.size(0)))
        perm = torch.tensor(perm)
        perm = perm.to(torch.long)
        neg_row_hetro, neg_col_hetro = neg_row_hetro[perm], neg_col_hetro[perm]

        neg_adj_mask[neg_row_hetro, neg_col_hetro] = 0
        test_neg_edge_index = torch.stack(
            [neg_row_hetro, neg_col_hetro], dim=0)

        return neg_adj_mask, test_neg_edge_index

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    # TODO: not sure it's right... might need to be deleted
    row, col = get_upper_triangular_portion(row, col)

    row, col = randomize_edges_order(row, col)
    (row_hetro, col_hetro), (row_homo, col_homo) = \
        get_hetro_homo_split(row, col, data.y)

    num_test = int(math.floor(test_ratio * row_hetro.shape[0]))

    # Positive edges.
    data.train_pos_edge_index, data.test_pos_edge_index = \
        get_pos_edges_split(row_hetro, col_hetro,
                            row_homo, col_homo, num_test)

    # Negative edges.
    data.train_neg_adj_mask, data.test_neg_edge_index = \
        get_neg_edges_split(num_nodes, row, col, y, num_test)

    return data

