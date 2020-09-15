import copy
import scipy.sparse as sp
import numpy as np
import time
from trial_manager import get_parser, TrialManager
from utils import DataUtils
import random

np.random.seed(12345)
random.seed(12345)

def main():
    args = get_parser().parse_args()
    for k, v in vars(args).items():
        print(f'{k:20}{v}')
    train(args)

def train_batch_command(tm):
    return tm.train_batch()

def train(args):
    graph_file = './data/%s/%s.npz' % (args.name, args.name)
    graph_file = graph_file.replace('.npz', '_train.npz')
    data_loader = DataUtils(graph_file)

    n = args.n_trials
    res_hom, res_het = [0] * n, [0] * n
    tm = [0] * n
    for i in range(n):
        tm[i] = TrialManager(args=copy.deepcopy(args), ind=i, data_loader=data_loader)
    import tensorflow
    tf = tensorflow.compat.v1

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)
    losses = []

    with sess.as_default():
        for b in range(1, args.num_batches + 1):
            fd = {}
            to_comp = []
            for to_comp1, fd1 in map(train_batch_command, tm):
                to_comp.extend(to_comp1)
                for k, v in fd1.items():
                    fd[k] = v
            res = sess.run(to_comp, feed_dict=fd)
            losses.append(res[0::2])
            if (b % 25) == 0:
                losses = np.array(losses)
                for i in range(n):
                    res, val_hom_auc = tm[i].test()
                    best_test_hom_auc, best_test_het_auc = res['hom'], res['het']
                    res_hom[i], res_het[i] = best_test_hom_auc * 100, best_test_het_auc * 100
                    print(
                        f'batch:{b:8} - '
                        f'time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} - '
                        f'loss:{np.mean(losses[:, i]):.4f} - '
                        f'val(hom):{val_hom_auc*100:.4f} - '
                        f'test(by best val):[hom:{best_test_hom_auc:.4f},het:{best_test_het_auc:.4f}]'
                    )
                losses = []
        print('finished')

    def stats(x):
        return f'{np.mean(x):.2f}, {np.std(x) / np.sqrt(len(x)):.2f}'

    print('hom', stats(res_hom), [f'{xx:.2f}' for xx in res_hom])
    print('het', stats(res_het), [f'{xx:.2f}' for xx in res_het])


if __name__ == '__main__':
    main()

