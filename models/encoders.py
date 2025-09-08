"""Graph encoders."""
import sys
sys.path.append('/data1/ziyan/RFN2')# Please change accordingly!

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath

from geoopt import ManifoldParameter


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.kp:
            #print(f"adj: {adj}")
            nei, nei_mask = adj
            input = (x, nei, nei_mask)
            output, _, __ = self.layers.forward(input)
            #Actually corresponds to (h, nei, nei_mask), but we only need h as output
        elif self.encode_graph:
            #Not sure what this means
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class HKPNet(Encoder):
    """
    HKPNet.
    """

    def __init__(self, c, args):
        super(HKPNet, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        if args.linear_before != None:
            self.before = True
            args.linear_before = int(args.linear_before)
            self.linear_before = hyp_layers.LorentzLinear(self.manifold, dims[0], args.linear_before, args.bias, args.dropout)
            dims[0] = args.linear_before
        else:
            self.before = False
        
        sample_ratio = getattr(args, 'frame_sample_ratio', None)
        
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.KPGraphConvolution(
                    self.manifold, args.kernel_size, args.KP_extent, args.radius, 
                    in_dim, out_dim, args.bias, args.dropout, 
                    nonlin=act, deformable=args.deformable, 
                    corr=args.corr, nei_agg=args.nei_agg, 
                    use_frame=args.use_frame, temperature=args.temperature,
                    sample_ratio=sample_ratio, use_learnable_rotation=args.use_learnable_rotation,
                    layer_idx=i
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.kp = True

    def encode(self, x, adj):
        if self.before:
            x = self.linear_before(x)
        return super(HKPNet, self).encode(x, adj)

