from argparse import ArgumentParser

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core import LightningModule
import torch.optim as optim

# This is under try so that it won't reshuffle the imports when refactoring.
#   Error in torch + sklearn, both use TSL (statis libraries)
#   and torch run out of space in the TSL if imported after sklearn.
try:
    import sklearn
    import sklearn.metrics
except:
    raise

from data import (STRINGDataset, custom_collate_fn, EqualProbabilityPerDatasetRandomBatchSampler,
                  homo_interactions, hetro_interactions, all_interactions)
from grad_utils import GradientReversal
from models import (PredictionHead, TAPEProteinBert, ProtTransBertModel,
                    ProtTransAlbertModel, ProtTransXLNetModel)
from functools import reduce
from collections import defaultdict, OrderedDict
from typing import List, Union
from torch.utils.data import ConcatDataset
from functools import partial

import pytorch_lightning as pl

pl.seed_everything(12345)


class PPIDomainAdaptation(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        attrbute_to_set = [
            'load_pretrained', 'co_th', 'da_coef', 'co_coef', 'lr', 'bb_lr',
            'data_root', 'true_prob', 'max_length', 'epoch_size',
            'epoch_size_val', 'batch_size', 'batch_size_val', 'num_workers',
            'bb', 'use_multipleheads', 'vv_coef', 'hh_coef'
        ]

        for attr in attrbute_to_set:
            setattr(self, attr, hparams[attr])

        def listdict(d: dict): return defaultdict(list, d)
        # list per loss. (Will always report everything either way)
        self.interactions_per_losses_per_split = {
            'train': listdict({'labeled': homo_interactions,
                               'da': all_interactions,
                               'consist': hetro_interactions, }),
            'val': listdict({'labeled': hetro_interactions})
        }

        self.exp_name = '_'.join([
            f'bb={hparams["bb"]}',
            f'acc={hparams["accumulate_grad_batches"]}',
            f'upt={int(self.load_pretrained)}',
            f'con=c{self.co_coef}.t{self.co_th}',
            f'dac={self.da_coef}',
            f'lr={self.lr}',
            f'blr={self.bb_lr}',
            f'tp={self.true_prob}',
            f'trn=bs.{self.batch_size}.es.{self.epoch_size}',
            f'val=bs.{self.batch_size_val}.es.{self.epoch_size_val}',
            f'nw={self.num_workers}',
            f'ml={self.max_length}',
            f'mh={int(self.use_multipleheads)}',
            f'vvc={self.vv_coef}',
            f'hhc={self.hh_coef}',
        ])

        # backbone
        bb = hparams['bb']
        if self.bb == 'tape':
            self.backbone = \
                TAPEProteinBert(load_pretrained=self.load_pretrained)
        elif self.bb in ['ptra_bert', 'ptra_albert', 'ptra_xlnet']:
            if self.bb == 'ptra_bert':
                bbcls = ProtTransBertModel
            elif self.bb == 'ptra_albert':
                bbcls = ProtTransAlbertModel
            elif self.bb == 'ptra_xlnet':
                bbcls = ProtTransXLNetModel
            self.backbone = bbcls(
                folder=hparams['data_root_prot_trans'],
                load_pretrained=self.load_pretrained)
        else:
            raise AttributeError(
                f'Unknown backbone {self.bb}.')

        # heads
        dim = self.backbone.hidden_size
        self.heads = nn.ModuleDict({
            'virusvirus': PredictionHead(dim=dim, no_layers=1, no_cls=2),
            'hosthost': PredictionHead(dim=dim, no_layers=1, no_cls=2),
            'da': nn.Sequential(
                GradientReversal(),
                PredictionHead(dim=dim, no_layers=1, no_cls=3))
        })

        if not self.use_multipleheads:
            self.heads['hosthost'] = self.heads['virusvirus']

    def forward(self, x: List[List[str]]):
        def truncate(seq):
            max_length = self.max_length - 2  # not including <s> and </s>
            if len(seq) > max_length:
                start_index = random.randint(0, len(seq) - max_length)
                end_index = start_index + max_length
                seq = seq[start_index: end_index]
                assert len(seq) == max_length
            return seq

        x = [[truncate(x1), truncate(x2)] for x1, x2 in x]

        x = self.backbone(x, max_length=self.max_length)
        x = {k: pred(x) for k, pred in self.heads.items()}
        x['virusvirus'] = x['virusvirus'] * self.vv_coef
        x['hosthost'] = x['hosthost'] * self.hh_coef
        v = x['virusvirus_lsm'] = F.log_softmax(x['virusvirus'], dim=1)
        h = x['hosthost_lsm'] = F.log_softmax(x['hosthost'], dim=1)
        x['virushost'] = ((v.exp() + h.exp()) * 0.5).log()
        return x

    def _losses(self, preds, y, interactions):
        def label_loss(preds, y, interactions):
            preds = [preds[k] for k in all_interactions]
            preds = torch.stack(preds, dim=1)
            idx = interactions.view(-1, 1, 1).expand(-1, 1, 2)
            preds = torch.gather(preds, dim=1, index=idx)[:, 0, :]
            return {
                'ce': F.cross_entropy(preds, y, reduction='none'),
                'acc': (preds.argmax(dim=1) == y).float(),
                'pred': preds[:, 1].detach(),
            }

        def da_loss(preds, interactions):
            return {
                'da': F.cross_entropy(preds['da'], interactions, reduction='none'),
                'da_acc': (preds['da'].argmax(dim=1) == interactions).float()
            }

        def consist_loss(pred):
            def ce(p, q): return -(p.detach().exp() * q).sum(dim=1)
            cons_v = ce(pred['virusvirus_lsm'], pred['hosthost_lsm'])
            cons_h = ce(pred['hosthost_lsm'], pred['virusvirus_lsm'])
            max_v = pred['virusvirus_lsm'].max(dim=1)[0]
            max_h = pred['hosthost_lsm'].max(dim=1)[0]
            mask_v = (max_v.exp() >= self.co_th).float()
            mask_h = (max_h.exp() >= self.co_th).float()
            return {
                'consist': ((cons_v*mask_v+cons_h*mask_h) / (mask_v+mask_h+1e-10)),
                'consist_mask': ((mask_v+mask_h) > 0).float()
            }

        return {
            **label_loss(preds, y, interactions),
            **da_loss(preds, interactions),
            **consist_loss(preds),
        }

    def _forward_losses(self, x: dict):
        seqs, y, interactions = x['seqs'], x['y'], x['interactions']
        preds = self.forward(seqs)
        losses = self._losses(preds=preds, y=y, interactions=interactions)
        losses.update({'y': y, 'interactions': interactions})
        return losses

    def training_step(self, x: dict, batch_idx):
        return self._forward_losses(x)

    def validation_step(self, x: dict, batch_idx):
        return self._forward_losses(x)

    def test_step(self, x: dict, batch_idx):
        return self._forward_losses(x)

    def _loss_and_stats(self, outputs: List[dict], splt: str):
        assert splt in ['train', 'val']

        all_keys = set([k for x in outputs for k in x.keys()])

        x = {k: [o[k] for o in outputs if k in o.keys()] for k in all_keys}
        x = {k: torch.cat(v) for k, v in x.items()}

        losses = []
        metric, pb_metric = OrderedDict(), OrderedDict()

        def add_metric(name: str, pb_name: str, value: Union[float, torch.Tensor]):
            if isinstance(value, float):
                value = torch.tensor(value)
            value = value.detach().cpu()
            metric[f'{splt}/{name}'] = value
            pb_metric[f'{splt[0]}/{pb_name}'] = f'{value.item():.3f}'

        def add_metrics(name: str, pb_name: str, values: dict):
            for k, v in values.items():
                add_metric(f'{name}/{k}', f'{pb_name}/{k[0]}', v)

        def add_loss(values, coef: float = 1.):
            if coef != 0:
                if 'loss' in values.keys():
                    losses.append(values['loss'] * coef)

        def add_metrics_and_loss(name: str, pb_name: str, values: dict, coef: float = 1.):
            add_loss(values, coef)
            add_metrics(name, pb_name, values)

        def get_mask(interactions):
            with torch.no_grad():
                if interactions == None or len(interactions) == 0:
                    return (x['interactions'] < 0).float()
                interactions = map(all_interactions.index, interactions)
                interactions_mask = map(
                    lambda i: x['interactions'] == i, interactions)
                return reduce(lambda a, b: a | b, interactions_mask).float()

        def supervised_loss_acc_auc(interactions):
            mask = get_mask(interactions)
            res = {}
            if mask.sum() != 0:
                res['loss'] = (x['ce'] * mask).sum() / (mask.sum())
                res['acc'] = (x['acc'] * mask).sum() / (mask.sum())
                if splt != 'train':
                    # AUC only relevant when done at end of epoch
                    res['AUC'] = torch.tensor(sklearn.metrics.roc_auc_score(
                        x['y'][mask != 0].detach().cpu(),
                        x['pred'][mask != 0].detach().cpu()))
            return res

        def da_loss_acc(interactions):
            mask = get_mask(interactions)
            if mask.sum().item() != 0:
                return {'loss': (x['da'] * mask).sum() / mask.sum(),
                        'acc': (x['da_acc'] * mask).sum() / mask.sum()}
            return {}

        def consist_loss(interactions):
            assert (len(interactions) == 0 or
                    (len(interactions) == 1 and interactions[0] == 'virushost'))
            mask = get_mask(interactions)
            mask = mask * x['consist_mask']
            if mask.sum().item() != 0:
                return {
                    'loss': (x['consist'] * mask).sum() / mask.sum()
                }
            return {}

        rel_interactions = self.interactions_per_losses_per_split[splt]

        main_sup = supervised_loss_acc_auc(rel_interactions['labeled'])
        add_metrics_and_loss('labeled', 'l', main_sup)

        da = da_loss_acc(rel_interactions['da'])
        add_metrics_and_loss('da', 'd', da, coef=self.da_coef)

        consist = consist_loss(rel_interactions['consist'])
        add_metrics_and_loss('consist', 'c', consist, coef=self.co_coef)

        # logs
        with torch.no_grad():
            for inter in all_interactions:
                inter_short = {'virusvirus': 'v',
                               'virushost': 'vh',
                               'hosthost': 'h'}[inter]
                sup = supervised_loss_acc_auc([inter])
                add_metrics(f'{inter}/labeled', f'{inter_short}/l', sup)

                da = da_loss_acc([inter])
                add_metrics(f'{inter}/da', f'{inter_short}/d', da)

        res = {'log': metric,
               'progress_bar': pb_metric}
        if len(losses) == 0:
            losses.append(x['consist'].sum()*.0)

        res['losses'] = losses

        if splt == 'train':
            res['loss'] = sum(losses)
        else:
            res[f'{splt}_loss'] = sum(losses)
            res[f'{splt}_acc'] = main_sup['acc']
            res[f'{splt}_auc'] = main_sup['AUC']
        return res

    def training_step_end(self, output, **kwargs):
        return self._loss_and_stats([output], 'train')

    def validation_epoch_end(self, outputs: dict):
        res = self._loss_and_stats(outputs, 'val')
        del outputs
        return res

    def test_epoch_end(self, outputs: dict):
        return self._loss_and_stats(outputs, 'val')

    def configure_optimizers(self):
        params = [
            {'params': self.backbone.parameters(), 'lr': self.bb_lr},
            {'params': self.heads.parameters(), 'lr': self.lr}
        ]

        opt = optim.Adam(params)
        return opt

    def _get_dataloader(self, splt: str):
        assert splt in ['train', 'val']

        def get_dataset(inter: str):
            return STRINGDataset(data_root=self.data_root, split=splt,
                                 interaction_type=inter, true_prob=self.true_prob)

        interactions = all_interactions if splt == 'train' else hetro_interactions
        datasets = [get_dataset(i) for i in interactions]
        epoch_size = self.epoch_size if splt == 'train' else self.epoch_size_val
        batch_size = self.batch_size if splt == 'train' else self.batch_size_val

        dataset = ConcatDataset(datasets)

        seed = 0 if splt != 'train' else None
        # when none it shuffle using his own seed

        batch_sampler = EqualProbabilityPerDatasetRandomBatchSampler(
            dataset=dataset, epoch_size=epoch_size, batch_size=batch_size, seed=seed)
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, pin_memory=True,
            collate_fn=custom_collate_fn, num_workers=self.num_workers)

    def prepare_data(self):
        _ = self._get_dataloader('train')
        _ = self._get_dataloader('val')

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        return self._get_dataloader('val')

    def test_dataloader(self):
        return self._get_dataloader('val')

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--bb', default='tape', type=str)
        parser.add_argument('--bb_lr', default=1e-6, type=float)

        parser.add_argument('--lr', default=2e-4, type=float)
        parser.add_argument('--true_prob', default=0.5, type=float)
        parser.add_argument('--num_workers', default=32, type=int)

        parser.add_argument('--da_coef', type=float, required=True)
        parser.add_argument('--co_coef', type=float, required=True)

        parser.add_argument('--co_th', default=0.0, type=float)
        parser.add_argument('--vv_coef', default=1., type=float)
        parser.add_argument('--hh_coef', default=1., type=float)
        parser.add_argument('--max_length', default=500, type=int)

        parser.add_argument('--train_from_scratch',
                            dest='load_pretrained', action='store_false')
        parser.add_argument('--dont_use_multipleheads',
                            dest='use_multipleheads', action='store_false')

        parser.add_argument('--batch_size', default=1,
                            type=int, help='batch size to use')
        parser.add_argument('--epoch_size',
                            default=3 * 8000 // 4, type=int)
        parser.add_argument('--batch_size_val', default=45,
                            type=int, help='batch size to use')
        parser.add_argument('--epoch_size_val', default=9000, type=int)
        parser.add_argument('--data_root',
                            default='/home/idokessler/mnt/g01/STRING', type=str)
        parser.add_argument('--data_root_prot_trans',
                            default='./pretrained_models/', type=str)
        return parser

    
def main(hparams):
    pl.seed_everything(12345)
    
    model = PPIDomainAdaptation(vars(hparams))
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(model)
    return trainer.callback_metrics['val/labeled/acc'].item()


    
def get_args():
    parent_parser = ArgumentParser(add_help=False)
    parser = PPIDomainAdaptation.add_model_specific_args(parent_parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        early_stop_callback=False,
        gpus=-1,
        benchmark=True
    )
    return parser.parse_args()


if __name__ == '__main__':
    print(main(get_args()))
