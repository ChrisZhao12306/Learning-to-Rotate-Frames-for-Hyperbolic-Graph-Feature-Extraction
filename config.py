import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.1, 'dropout probability'),
        'cuda': (1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (1000, 'maximum number of epochs to train for'),
        'weight_decay': (1e-3, 'l2 regularization strength, the lambda'),
        'optimizer': ('radam', 'which optimizer to use, can be any of [rsgd, radam, adam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (8, 'seed for training'),
        'log_freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval_freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (1, '1 to save model and logs and 0 otherwise'),
        'save_dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep_c': (0, ''),
        'lr_reduce_freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print_epoch': (True, ''),
        'grad_clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min_epochs': (100, 'do not early stop before min-epochs'),
        'print_grad': (False, 'whether to print gradient for every parameter')
    },
    'model_config': {
        'use_geoopt': (False, "which manifold class to use, if false then use basd.manifold"),
        'AggKlein':(True, "if false, then use hyperboloid centorid for aggregation"),
        'corr': (1,'0: Agg{x_ik,d(x_i ominus x, x_k)}, 1: Agg{x_ik,d(x_ik,x_k)}, 2:Agg{x_i ominus x,d(x_i ominus x, x_k)}'),
        'nei_agg': (2, '0: simple uniform weight midpoint/centroid, 1: attention based neighbor aggregation, 2: GIN based neighbor aggregation'),
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('HKPNet', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN, HyboNet,HKPNet]'),
        'dim': (32, 'embedding dimension'),
        'manifold': ('Lorentz', 'which manifold to use, can be any of [Euclidean, Hyperboloid, Lorentz]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'margin': (2., 'margin of MarginLoss'),
        'pretrained_embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos_weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num_layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('None', 'which activation function to use (or None for no activation)'),
        'n_heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double_precision': ('1', 'whether to use double precision'),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'kernel_size': (6, 'number of kernels'),
        'KP_extent': (0.66, 'influence radius of each kernel point'),
        'radius': (1, 'radius used for kernel point init'),
        'deformable': (False, 'deformable kernel'),
        'linear_before': (32, 'dim of linear before gcn'),
        'use_frame': (True, 'whether to use frame kernel points for Lorentz model'),
        'scale_j': (0, 'scale parameter j for frame kernel points'),
        'temperature': (1.0, 'temperature parameter for controlling rotation softness in frame kernels'),
        'frame_sample_ratio': (1.0, 'sampling ratio for frame kernel points (0-1), 1.0 for no sampling'),
        'use_learnable_rotation': (True, 'whether to use learnable rotation matrix in frame method')
    },
    'data_config': {
        'dataset': ('cornell', 'which dataset to use'),
        'batch_size': (32, 'batch size for gc'),
        'val_prop': (0.05, 'proportion of validation edges for link prediction'),
        'test_prop': (0.1, 'proportion of test edges for link prediction'),
        'use_feats': (1, 'whether to use node features or not'),
        'normalize_feats': (1, 'whether to normalize input node features'),
        'normalize_adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split_seed': (1234, 'seed for data splits (train/test/val)'),
        'split_graph': (False, 'whether to split the graph')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
