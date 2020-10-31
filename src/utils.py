import numpy as np
from collections import deque
import argparse

# compute moving average
def moving_average(x, n):
    # create a deque object
    x_deque = deque(maxlen=n)
    ma_x = []
    for x_ in x:
        x_deque.append(x_)
        ma_x.append(np.mean(x_deque))
    return np.array(ma_x)

def parse_args():
    """
    This function parses the input arguments
    """
    # set up input arguments 
    parser = argparse.ArgumentParser(description='Setup arameters as arguments')
    parser.add_argument('--n_way', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='number of samples per class')
    parser.add_argument('--meta_train_k_shot', type=int, default=-1,
                        help='number of samples per class during meta-training')
    parser.add_argument('--inner_update_lr', type=float, default=0.4,
                        help='learning rate for inner update loop')
    parser.add_argument('--num_inner_updates', type=int, default=1,
                        help='number of inner updates per outer update')
    parser.add_argument('--meta_batch_size', type=int, default=25,
                        help='meta batch size')
    parser.add_argument('--meta_train_batch_size', type=int, default=-1,
                        help='meta batch size during meta-training')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='meta learning rate for outer loop')
    parser.add_argument('--num_filters', type=int, default=32,
                        help='filter size for MAML architecture')
    parser.add_argument('--learn_inner_update_lr', default=False, action='store_true',
                        help='learn learning rate for inner updates')
    parser.add_argument('--modeldir', type=str, default='./models',
                        help='model directory for saving trained models')
    parser.add_argument('--logdir', type=str, default='./logs',
                        help='logs directory for generating CSV logs')
    parser.add_argument('--plotdir', type=str, default='./plots',
                        help='directory for generating plots')
    parser.add_argument('--dataset', type=str, default='omniglot',
                        help='dataset', choices=['omniglot', 'pose'])
    parser.add_argument('--data_path', type=str, default='./datasets/omniglot_resized',
                        help='data path')
    parser.add_argument('--meta_train_iterations', type=int, default=10000,
                        help='number of metatraining iterations')
    parser.add_argument('--mutual_exclusive', default=False, action='store_true',
                        help='whether or not to prepare data in a mutual exclusive manner')

    return parser.parse_args()