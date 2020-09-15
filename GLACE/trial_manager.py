import argparse

from model import GLACE
from utils import score_link_prediction
from utils import sparse_feeder

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', default='cora_ml', type=str)
    parser.add_argument('--suf', default='', type=str)
    parser.add_argument('--proximity', default='first-order', type=str, help='first-order or second-order')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--learning_rate_bb', default=0.001, type=float)
    parser.add_argument('--num_batches', type=int, default=1500)
    parser.add_argument('--da_coef', type=float, default=0.1)
    parser.add_argument('--co_coef', type=float, default=0.1)
    parser.add_argument('--classes', type=int, nargs='+', default=-1)
    parser.add_argument('--use_multihead', action='store_true')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--n_trials', type=int, default=16)

    return parser


class TrialManager:
    def __init__(self, args, ind, data_loader):
        self.ind = ind
        args.X = data_loader.X if args.suf != 'oh' else sp.identity(data_loader.X.shape[0])
        import tensorflow
        tf = tensorflow.compat.v1
        tf.disable_v2_behavior()
        tf.set_random_seed(12345)
        args.X_tf = tf.SparseTensor(*sparse_feeder(args.X))
        num_batches = args.num_batches
        args.labels = data_loader.labels
        args.val_edges = data_loader.val_edges
        args.val_ground_truth = data_loader.val_ground_truth
        args.test_edges = data_loader.test_edges
        args.test_ground_truth = data_loader.test_ground_truth
        self.model = GLACE(args, ind)
        self.args = args
        self.best_val_scores = {'hom': 0}
        self.best_test_scores = {'het': 0, 'hom': 0}
        self.data_loader = data_loader

    def train_batch(self):
        import tensorflow
        tf = tensorflow.compat.v1
        tf.disable_eager_execution()
        args = self.args
        model = self.model
        u_i, u_j, label, is_hom = self.data_loader.fetch_next_batch(batch_size=args.batch_size, K=args.K,
                                                                    labels_to_use=args.classes)
        feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.is_hom: is_hom}
        return [model.loss, model.train_op], feed_dict

    def test(self):
        import tensorflow
        tf = tensorflow.compat.v1
        tf.disable_eager_execution()
        sess = tf.get_default_session()
        model = self.model
        val_energy_hom, val_energy_het = sess.run([model.neg_val_energy_hom, model.neg_val_energy_het])
        val_hom_auc, _ = score_link_prediction(model.val_ground_truth_hom, val_energy_hom, 'hom_val')
        val_het_auc, _ = score_link_prediction(model.val_ground_truth_het, val_energy_het, 'het_val')

        if self.best_val_scores['hom'] < val_hom_auc:
            self.best_val_scores['hom'] = val_hom_auc
            test_energy_hom, test_energy_het = sess.run(
                [model.neg_test_energy_hom, model.neg_test_energy_het])
            self.best_test_scores['hom'], _ = score_link_prediction(model.test_ground_truth_hom, test_energy_hom, 'hom_test')
            self.best_test_scores['het'], _ = score_link_prediction(model.test_ground_truth_het, test_energy_het, 'het_test')

        return self.best_test_scores, val_hom_auc

