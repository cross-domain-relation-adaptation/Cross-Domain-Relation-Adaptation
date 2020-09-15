from losses import ConsistencyLoss, DomainAdaptationLoss, EmbeddingLoss
from params import Params
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import SGConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from grad_utils import rev_grad
import torch.nn.functional as F
from graph import Graph
from torch import Tensor as T


class GCNDomainAdaptation(nn.Module):
    def __init__(self, graph: Graph, params: Params):
        super().__init__()
        self.params = params
        self.conv = GCNConv(params.hidden_c, graph.no_cls)
        self.loss = DomainAdaptationLoss(params)

    def forward(self, g: Graph, hiddens: T, q_edge_index: T) -> Dict[str, T]:
        hiddens = rev_grad(hiddens)
        emb_da = self.conv(hiddens, g.edge_for_gnn)

        return self.loss.forward(pred_da=emb_da, y=g.y)


class GNNModel(nn.Module):
    def __init__(self,
                 graph: Graph,
                 params: Params):
        super().__init__()
        self.multiheads = params.multiheads
        self.hidden_c = params.hidden_c
        self.out_c = params.out_c
        self.no_heads = graph.no_cls if self.multiheads else 1

        n, num_features = graph.x.shape
        self.conv1 = GCNConv(num_features, self.hidden_c)
        self.conv2 = GCNConv(self.hidden_c, self.out_c * self.no_heads)

        # for correct initialization
        weights = [GCNConv(self.hidden_c, self.out_c).weight
                   for _ in range(self.no_heads)]
        weights = torch.cat(weights, dim=-1)
        self.conv2.weight = nn.Parameter(weights)

        self.embedding_loss = EmbeddingLoss(params=params)
        self.da = GCNDomainAdaptation(graph=graph, params=params)
        self.consistency_loss = ConsistencyLoss(params=params)

    def get_hiddens(self, g: Graph) -> T:
        x = self.conv1(g.x, g.edge_for_gnn)
        x = F.relu(x)
        return x

    def get_concat_hiddens(self, g: Graph, q_edge_index: T) -> T:
        x = self.get_hiddens(g=g)
        q_row, q_col = q_edge_index
        return torch.cat([x[q_row], x[q_col]], dim=-1)

    def get_embeddings(self, hiddens: T, g: Graph) -> T:
        x = self.conv2(hiddens, g.edge_for_gnn)
        return x.view(x.shape[0], self.no_heads, self.out_c)

    def get_heads_preds(self, emb: T, q_edge_index: T) -> T:
        q_row, q_col = q_edge_index
        return torch.einsum('ehf, ehf -> eh', emb[q_row], emb[q_col])

    def extract_relevant_preds(self, preds: T, g: Graph,  q_edge_index: T) -> Tuple[T, T]:
        def get_preds(heads: T) -> T:
            heads = heads.unsqueeze(1)
            return preds.gather(dim=1, index=heads)[:, 0]

        q_row, q_col = q_edge_index
        y = g.y if self.multiheads else torch.zeros_like(g.y)
        return get_preds(y[q_row]), get_preds(y[q_col])

    def forward(self, g: Graph, q_edge_index: T) -> Dict[str, T]:
        hiddens = self.get_hiddens(g)
        emb = self.get_embeddings(hiddens=hiddens, g=g)
        preds = self.get_heads_preds(emb=emb, q_edge_index=q_edge_index)
        pred_i, pred_j = self.extract_relevant_preds(
            preds=preds, g=g, q_edge_index=q_edge_index)
        pred = (pred_i + pred_j)*.5 if self.multiheads else pred_i

        return {
            'hiddens': hiddens,
            'emb': emb,
            'pred_i': pred_i,
            'pred_j': pred_j,
            'pred': pred,
        }

    def forward_losses(self, res: Dict[str, T], g: Graph, q_edge_index: T, q_edge_y: T) -> Dict[str, T]:
        return {**self.da.forward(g=g,
                                  hiddens=res['hiddens'],
                                  q_edge_index=q_edge_index),
                **self.consistency_loss.forward(g=g,
                                                q_edge_index=q_edge_index,
                                                pred_i=res['pred_i'],
                                                pred_j=res['pred_j']),
                **self.embedding_loss.forward(g=g, q_edge_index=q_edge_index,
                                              link_pred=res['pred'], link_y=q_edge_y)}

