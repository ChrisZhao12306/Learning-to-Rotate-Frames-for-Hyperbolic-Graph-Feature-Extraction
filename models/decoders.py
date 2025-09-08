"""Graph decoders."""
import sys
sys.path.append('/data1/ziyan/RFN2')# Please change accordingly!

import math
import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import GraphConvolution, Linear

from layers.hyp_layers import LorentzLinear as LLinear

from geoopt import ManifoldParameter as geoopt_ManifoldParameter
from manifolds.base import ManifoldParameter as base_ManifoldParameter


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, 0, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class LorentzDecoder(Decoder):
    """
    MLP Decoder for HKPNet node classification models.
    """
    def __init__(self, c, args):
        super(LorentzDecoder, self).__init__(c)
        self.use_HCDist = True
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        if self.use_HCDist:
            self.cls = geoopt_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim)), manifold=self.manifold)
            if args.bias:
                self.bias = nn.Parameter(torch.zeros(args.n_classes))
        else:
            self.cls = LLinear(self.manifold, self.input_dim, self.output_dim, self.use_bias, args.dropout, nonlin=None)
        self.decode_adj = False

    def decode(self, x, adj):
        if self.use_HCDist:
            return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias
        else:
            return self.manifold.logmap0(self.cls(x))

#Decoders for node classification
model2decoder = {
    'HKPNet': LorentzDecoder,
}
