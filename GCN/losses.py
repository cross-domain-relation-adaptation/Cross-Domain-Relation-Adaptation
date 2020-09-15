from pytorch_lightning.metrics.classification import Accuracy
from torch.nn.modules.loss import CrossEntropyLoss
from graph import Graph
from typing import Dict
from params import Params
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch import Tensor as T

from pytorch_lightning.metrics import AUROC


class SoftMaxWrapper(nn.Module):
    def __init__(self, metric, multiclass: bool) -> None:
        super().__init__()
        self.multiclass = multiclass
        self.metric = metric

    def forward(self, pred: T, y: T):
        if self.multiclass:
            assert len(pred.shape) == 2
            pred = pred.argmax(dim=1)
        else:
            assert len(pred.shape) == 1
            pred = pred > 0
        return self.metric(pred.long(), y.long())


class MetricsHelper(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics_singleclass = nn.ModuleDict({
            'loss': BCEWithLogitsLoss(),
            'acc': SoftMaxWrapper(Accuracy(), multiclass=False),
            'auc': AUROC(),
        })
        self.metrics_multiclass = nn.ModuleDict({
            'loss': CrossEntropyLoss(),
            'acc': SoftMaxWrapper(Accuracy(), multiclass=True),
        })

    def forward(self, prefix: str, pred: T, y: T, multiclass: bool = False) -> Dict[str, T]:
        metrics = self.metrics_multiclass if multiclass else self.metrics_singleclass
        return {
            f'{prefix}/{mname}': metric(pred, y)
            for mname, metric in metrics.items()
        }


class Loss(nn.Module):
    def __init__(self, params: Params):
        super().__init__()
        self.params = params
        self.metrics = MetricsHelper()


class EmbeddingLoss(Loss):
    def forward(self, g: Graph, q_edge_index: T, link_pred: T, link_y: T) -> Dict[str, T]:
        res = {}
        q_row, q_col = q_edge_index
        mask_hetro = g.y[q_row] != g.y[q_col]

        cuts = {'all': (mask_hetro | True),
                'heterogeneous': mask_hetro,
                'homogeneous': ~mask_hetro, }
        for prefix, mask in cuts.items():
            pred = link_pred[mask]
            y = link_y[mask]
            res.update(self.metrics(prefix, pred, y.float()))

        return res


class DomainAdaptationLoss(Loss):
    def forward(self, pred_da: T, y: T) -> Dict[str, T]:
        return self.metrics('da', pred_da, y.long(), multiclass=True)


class ConsistencyLoss(Loss):
    def forward(self, g: Graph, q_edge_index: T, pred_i: T, pred_j: T) -> Dict[str, T]:
        y = g.y
        q_row, q_col = q_edge_index
        mask_hetro = y[q_row] != y[q_col]
        loss = (pred_i - pred_j)[mask_hetro].pow(2).mean()
        return {'consistency/loss': loss}

