"""Base model class."""
import sys
sys.path.append('/data1/ziyan/RFN2')# Please change accordingly!

# from os import POSIX_FADV_NOREUSE
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1, MarginLoss


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold

        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device) #Base model carries the curvature on cuda
        else:
            self.c = nn.Parameter(torch.Tensor([1.])) #Without specific indication, it's on cpu

        self.manifold = getattr(manifolds, self.manifold_name)()#Initialize a manifold, eg PoincareBall

        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            if not hasattr(args, 'feat_dim'):
                args.feat_dim = args.data['features'].shape[1]
            args.feat_dim = args.feat_dim + 1

        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)#Initialize an encoder, eg BKNet

    def encode(self, x, adj):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)#Using geoopt
        elif self.manifold.name in ['PoincareBall']:
            x = self.manifold.expmap0(x,c=self.c)#Using manifold.base
        h = self.encoder.encode(x, adj)
        #Note: h is the updated feature points matrix of shape (n,d')
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)#Initialize the BaseModel #This is inheritence!
        self.decoder = model2decoder[args.model](self.c, args)# This is composition!
        self.margin = args.margin
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return output[idx]

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        # print(data['labels'][idx].shape, data['labels'].shape)
        if self.manifold_name == 'Lorentz':
            #Margin Loss
            #correct = output.gather(1, data['labels'][idx].to(torch.long).unsqueeze(-1))
            #loss = F.relu(self.margin - correct + output).mean()
            #CE Loss
            loss = F.cross_entropy(output, data['labels'][idx].to(torch.long), self.weights.to(output.dtype))
        elif self.manifold_name == 'PoincareBall':
            #Margin Loss
            #correct = output.gather(1, data['labels'][idx].to(torch.long).unsqueeze(-1))
            #loss = F.relu(self.margin - correct + output).mean()
            #CE Loss
            loss = F.cross_entropy(output, data['labels'][idx].to(torch.long), self.weights.to(output.dtype))
        else:
            loss = F.cross_entropy(output, data['labels'][idx], self.weights)
            #loss = F.cross_entropy(output, data['labels'][idx].to(torch.long), self.weights.to(torch.float64))
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics
    
    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


        