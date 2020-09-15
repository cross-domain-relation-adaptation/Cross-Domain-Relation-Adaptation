from params import Params
import torch
import torch.optim
import torch.utils.data
from models import GNNModel
import pytorch_lightning as pl
from argparse import ArgumentParser
from graph import Graph

pl.seed_everything(12345)

class LightningGNNModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = Params(**kwargs)
        params = self.params
        self.save_hyperparameters(params._asdict())

        self.graph = Graph(params=params)
        self.model = GNNModel(graph=self.graph, params=params)

    def forward(self, q_edge_index: torch.Tensor):
        return self.model.forward(g=self.graph, q_edge_index=q_edge_index)['pred']

    def forward_and_losses(self, q_edge_index: torch.Tensor, link_y: torch.Tensor):
        res = self.model.forward(g=self.graph, q_edge_index=q_edge_index)
        metrics = self.model.forward_losses(
            res=res, g=self.graph, q_edge_index=q_edge_index, q_edge_y=link_y)

        return metrics

    def training_step(self, batch, batch_idx):
        params = self.params
        q_edge_index, link_y = self.graph.get_batch('train')
        metrics = self.forward_and_losses(q_edge_index, link_y)

        cls_loss = 'all/loss' if params.supervised else 'homogeneous/loss'
        loss = \
            metrics[cls_loss] + \
            metrics['da/loss'] * params.da_coef + \
            metrics['consistency/loss'] * params.co_coef

        metrics = {f'train/{k}': v for k, v in metrics.items()}
        return {'loss': loss,
                'log': metrics}

    def val_test_step(self, splt):
        is_test = {'val': False, 'test': True}[splt]

        q_edge_index, link_y = self.graph.get_batch(splt)
        metrics = self.forward_and_losses(q_edge_index, link_y)
        metrics = {f'{splt}/{k}': v for k, v in metrics.items()}
        checkpoint_on = None

        if not is_test:
            checkpoint_on = metrics['val/heterogeneous/auc']

        res: dict = {'log': {**metrics}}
        if checkpoint_on is not None:
            # Default checkpoint uses min (expect loss). so * -1
            checkpoint_on = checkpoint_on * -1
            res[f'{splt}_loss'] = checkpoint_on

        return res

    def validation_step(self, batch, batch_idx):
        return self.val_test_step('val')

    def test_step(self, batch, batch_idx):
        return self.val_test_step('test')

    def validation_epoch_end(self, outputs):
        assert len(outputs) == 1
        return outputs[0]

    def test_epoch_end(self, outputs):
        assert len(outputs) == 1
        return outputs[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.params.lr)
        return [optimizer], []

    @ staticmethod
    def _placeholder_batch_dataloader(n_len: int):
        dataset = torch.utils.data.TensorDataset(torch.Tensor(n_len))
        return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    def train_dataloader(self):
        return self._placeholder_batch_dataloader(self.params.epoch_size)

    def val_dataloader(self):
        return self._placeholder_batch_dataloader(1)

    def test_dataloader(self):
        return self._placeholder_batch_dataloader(1)

    @ staticmethod
    def add_model_specific_args(parent_parser):
        """Define parameters that only apply to this model"""
        def bool_(s: str): return s.lower() in ['true', '1', 'yes']

        parser = ArgumentParser(parents=[parent_parser])
        for param_name, param_type, param_default in Params.get_params_type_default():
            assert param_type in [bool, int, float, str], param_name
            if param_type == bool:
                param_type = bool_  # handle bool('False') == True
            parser.add_argument(f'--{param_name}',
                                default=param_default,
                                type=param_type)

        return parser


def main(args) -> dict:
    """ Main training routine specific for this project. """
    model = LightningGNNModel(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
    for k,v in trainer.callback_metrics.items():
        print(f'{k:30}{v}')
    trainer.test()


def run_cli():
    parent_parser = ArgumentParser(add_help=False)

    parser = pl.Trainer.add_argparse_args(parent_parser)
    parser = LightningGNNModel.add_model_specific_args(parser)

    parser.set_defaults(gpus=[1], max_epochs=10, num_sanity_val_steps=0,)
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
