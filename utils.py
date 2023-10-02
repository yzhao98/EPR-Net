import argparse
import json
import random
from datetime import datetime

import numpy as np
import torch


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),
        'args': str(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f)


def parse_args():
    parser = argparse.ArgumentParser()

    # problem
    parser.add_argument('--problem_id', default=0, type=int)
    parser.add_argument('--dimension', default=2, type=int)
    parser.add_argument('--D', default=0.1, type=float)
    parser.add_argument('--boundary_type', default='common_j', type=str)
    parser.add_argument('--resimulate', default=False, action='store_true')
    parser.add_argument('--x_max', default=3.0, type=float)
    parser.add_argument('--x_min', default=0.0, type=float)
    parser.add_argument('--delta_t', default=0.1, type=float)
    parser.add_argument('--enh_scale', default=1., type=float)
    parser.add_argument('--sample_size', default=10000, type=int)

    # loss parameters
    parser.add_argument('--rho_1', default=0.1, type=float)
    parser.add_argument('--rho_2', default=0.5, type=float)

    # training
    parser.add_argument('--num_potential_epochs', default=100, type=int)
    parser.add_argument('--num_force_epochs', default=100, type=int)
    parser.add_argument('--hidden_sizes', type=str)
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--nf', default=False, action='store_true')
    parser.add_argument('--dr', default=False, action='store_true')

    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--save_interval', default=10, type=int)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--load_model_path', default='', type=str)
    parser.add_argument('--model_save_dir', default='checkpoints', type=str)
    parser.add_argument('--save_ckpt', default=False, action='store_true')
    parser.add_argument('--use_gpu', default=False, action='store_true')

    args = parser.parse_args()

    # CUDA support
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    return args


def save_model(dnn, model_save_dir):
    torch.save(dnn.state_dict(), model_save_dir)


def load_model(dnn, model_load_dir):
    dnn.load_state_dict(torch.load(model_load_dir))
