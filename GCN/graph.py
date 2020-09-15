import math
from params import Params
from typing import Tuple
import torch
import torch.nn
from pathlib import Path
from torch_geometric.datasets import Planetoid, CitationFull
import torch_geometric.transforms


class Graph(torch.nn.Module):
    x: torch.Tensor
    y: torch.Tensor
    train_edge_index: torch.Tensor
    val_edge_index: torch.Tensor
    test_edge_index: torch.Tensor
    edge_for_gnn: torch.Tensor

    def __init__(self, params: Params):
        super().__init__()
        from torch import Tensor as T
        self.params = params

        dataset_type, dataset = params.dataset.split('_')
        path = Path(params.data_root) / f'data_{dataset_type}' / dataset

        assert dataset_type in ['planetoid', 'citation'], dataset_type
        dataset_type = Planetoid if dataset_type == 'planetoid' else CitationFull

        transform = torch_geometric.transforms.NormalizeFeatures()
        ds = dataset_type(path, dataset, transform=transform)
        assert len(ds) == 1, 'Only one slice should be available'

        data = ds[0]
        data = Graph._train_test_split_edges(data)

        x = data.x
        y = data.y
        train_edge_index: T = data.train_pos_edge_index
        val_edge_index: T = torch.stack([data.val_pos_edge_index,
                                         data.val_neg_edge_index], dim=0)
        test_edge_index: T = torch.stack([data.test_pos_edge_index,
                                          data.test_neg_edge_index], dim=0)

        if params.supervised:
            edge_for_gnn: T = train_edge_index
        else:
            row, col = train_edge_index
            mask_homo = y[row] == y[col]
            row_homo, col_homo = row[mask_homo], col[mask_homo]
            train_edge_index_homo = torch.stack([row_homo, col_homo], dim=0)
            edge_for_gnn: T = train_edge_index_homo

        # pytorch 1.6+ persistent
        self.register_buffer('x', x)
        # , persistent=False)
        self.register_buffer('y', y)
        # , persistent=False)
        self.register_buffer('train_edge_index', train_edge_index)
        # , persistent=False)
        self.register_buffer('val_edge_index', val_edge_index)
        # , persistent=False)
        self.register_buffer('test_edge_index', test_edge_index)
        # , persistent=False)
        self.register_buffer('edge_for_gnn', edge_for_gnn)
        # , persistent=False)

        self.no_cls = int(self.y.max().item()) + 1
        self.n, self.num_features = self.x.shape

    @torch.no_grad()
    def get_batch(self, splt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        from torch_geometric.utils import (
            negative_sampling, remove_self_loops, add_self_loops)
        n = self.x.shape[0]

        if splt == 'train':
            pos_edge_index = self.train_edge_index
            num_neg_edges = pos_edge_index.shape[1]

            pos_edge_clean, _ = remove_self_loops(pos_edge_index)
            pos_edge_w_self_loop, _ = add_self_loops(
                pos_edge_clean, num_nodes=n)

            neg_edge_index = negative_sampling(
                edge_index=pos_edge_w_self_loop,
                num_nodes=n,
                num_neg_samples=num_neg_edges)
        elif splt == 'val':
            pos_edge_index, neg_edge_index = self.val_edge_index
        elif splt == 'test':
            pos_edge_index, neg_edge_index = self.test_edge_index
        else:
            raise ValueError(f'Unknown splt: {splt}')

        query_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        link_y = torch.zeros_like(query_edge_index[0], dtype=torch.float)
        link_y[:pos_edge_index.shape[1]] = 1

        return query_edge_index, link_y

    def print_details(self) -> None:
        @torch.no_grad()
        def print_dist(prefix: str, edges):
            y = self.y
            row, col = edges
            n_total = row.shape[0]
            n_homo = (y[row] == y[col]).long().sum()
            n_hetro = n_total - n_homo
            print(f'{prefix:30}{n_total:10}{n_homo:10}{n_hetro:10}')

        print(f'{"name":30}{"#Total":10}{"#Homo":10}{"#Hetro":10}')
        print_dist('edge_for_gnn', self.edge_for_gnn)
        print_dist('train_edge_index', self.train_edge_index)
        print_dist('val_edge_index_pos', self.val_edge_index[0])
        print_dist('val_edge_index_neg', self.val_edge_index[1])
        print_dist('test_edge_index_pos', self.test_edge_index[0])
        print_dist('test_edge_index_neg', self.test_edge_index[1])

    @staticmethod
    def _train_test_split_edges(
            data,
            random_seed=12345):

        import pytorch_lightning as pl
        from torch_geometric.utils import train_test_split_edges
        import random
        import numpy as np
        import torch

        rand_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        pl.seed_everything(random_seed)
        data = train_test_split_edges(data)

        random.setstate(rand_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

        return data

