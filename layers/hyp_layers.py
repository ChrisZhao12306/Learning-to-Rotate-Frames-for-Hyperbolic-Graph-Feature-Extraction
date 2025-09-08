"""Hyperbolic layers."""
import sys
sys.path.append('/data1/ziyan/RFN2')# Please change accordingly!

import math
from numpy import dtype
from sklearn import manifold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt
from kernels.kernel_points import load_kernels

from geoopt import ManifoldParameter

def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias, scale=10)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

class LorentzMultiHeadedAttention(nn.Module):
    """
    Hyperbolic Multi-headed Attention
    """

    def __init__(self, head_count, dim, manifold, dropout = 0.0):
        super(LorentzMultiHeadedAttention, self).__init__()
        self.dim_per_head = dim // head_count
        self.dim = dim
        self.manifold = manifold
        self.head_count = head_count

        self.linear_key = LorentzLinear(manifold, dim, dim, dropout=dropout)
        self.linear_value = LorentzLinear(manifold, dim, dim, dropout=dropout)
        self.linear_query = LorentzLinear(manifold, dim, dim, dropout=dropout)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor([math.sqrt(dim)]))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, key, value, query, mask = None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        # key_len = key.size(1)
        # query_len = query.size(1)

        def shape(x):
            """Projection."""
            if len(x.size()) == 3:
                x = x.view(batch_size, -1, head_count, dim_per_head)
            return x.transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).view(batch_size, -1, head_count * dim_per_head)

        query = self.linear_query(query)
        key = self.linear_key(key)
        value =  self.linear_value(value)
        key = shape(key)
        value = shape(value)
        query = shape(query)
        # key_len = key.size(2)
        # query_len = query.size(2)

        attn = (2 + 2 * self.manifold.cinner(query, key)) / self.scale + self.bias
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            attn = attn.masked_fill(mask, -1e18)
        context = self.manifold.mid_point(value, attn)
        context = unshape(context)

        return context

class LorentzCentroidDistance(nn.Module):
    """
    Hyerbolic embeddings to Euclidean distance
    """

    def __init__(self, dim, n_classes, manifold, bias = True):
        super(LorentzCentroidDistance, self).__init__()
        self.manifold = manifold
        self.input_dim = dim
        self.output_dim = n_classes
        self.use_bias = bias
        self.cls = ManifoldParameter(
            self.manifold.random_normal((n_classes, dim), std=1./math.sqrt(dim)), 
            manifold=self.manifold)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_classes))
        
    def forward(self, x):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias

class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        # print(type(self.in_features), type(self.out_features), self.bias)
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        #print(self.nonlin)
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1 + 1e-4
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)

# We wanna use it in neighbor aggregation via GIN idea
#LorentzLinear(manifold, out_channels+out_channels, 1, use_bias, dropout, nonlin=None)
class LMLP(nn.Module):
    def __init__(self, manifold, in_features, out_features, use_bias, dropout, nonlin):
        super(LMLP, self).__init__()
        self.linear1 = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin = nonlin)

        self.linear2 = LorentzLinear(manifold, out_features, out_features, use_bias, dropout, nonlin = None)

    def forward(self, x_nei_transform):
        #x_nei_transform: (n, nei_num, d')
        h = self.linear1.forward(x_nei_transform)
        return self.linear2.forward(h)


class LorentzAgg(Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            # self.att = DenseAtt(in_features, dropout)
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        # x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                query = self.query_linear(x)
                key = self.key_linear(x)
                att_adj = 2 + 2 * self.manifold.cinner(query, key)
                att_adj = att_adj / self.scale + self.bias
                att_adj = torch.sigmoid(att_adj)
                att_adj = torch.mul(adj.to_dense(), att_adj)
                support_t = torch.matmul(att_adj, x)
            else:
                adj_att = self.att(x, adj)
                support_t = torch.matmul(adj_att, x)
        else:
            support_t = torch.spmm(adj, x)
        # output = self.manifold.expmap0(support_t, c=self.c)
        denom = (-self.manifold.inner(None, support_t, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        output = support_t / denom
        return output

    def attention(self, x, adj):
        pass

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
    
    
    
    
def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')

class KernelPointAggregation(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, KP_extent, radius,
                 manifold, use_bias, dropout, nonlin=None,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, corr=0, nei_agg=0, modulated=False, use_frame=False, temperature=1.0, sample_ratio=None, use_learnable_rotation=True):
        super(KernelPointAggregation, self).__init__()
        # Save parameters
        self.manifold = manifold
        self.K = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.corr = corr
        self.nei_agg = nei_agg
        self.use_frame = use_frame
        self.temperature = temperature
        self.sample_ratio = sample_ratio
        self.use_learnable_rotation = use_learnable_rotation
        self.layer_idx = 0  

        # Initialize kernel points
        self.kernel_points, self.rotation = None, None
        self.init_KP()
        
        self.K = self.kernel_points.shape[0]

        # Initialize weights
        print("nonlin in LorentzLinear:", nonlin)
        self.linears = nn.ModuleList([LorentzLinear(manifold, in_channels, out_channels, use_bias, dropout, nonlin=nonlin)
                                    for _ in range(self.K)])

        if self.nei_agg == 0:
            pass
        elif self.nei_agg == 1:
            self.atten1 = LorentzLinear(manifold, out_channels+out_channels, 1, use_bias, dropout, nonlin=None)
            self.atten2 = LorentzLinear(manifold, out_channels+out_channels, 1, use_bias, dropout, nonlin=None)
        elif self.nei_agg == 2:
            self.MLP_f = LMLP(manifold, out_channels, 2*out_channels, use_bias, dropout, nonlin=nonlin)
            self.MLP_fi = LMLP(manifold, 2*out_channels, out_channels, use_bias, dropout, nonlin=None)
        else:
            raise NotImplementedError("The specified correlation type is not implemented.")

        if deformable:
            self.offset_dim = (self.in_channels - 1) * self.K + 1
            self.offset_conv = KernelPointAggregation(self.K,
                                    self.in_channels,
                                    self.offset_dim,
                                    KP_extent,
                                    radius,
                                    self.manifold,
                                    use_bias,
                                    dropout,
                                    fixed_kernel_points=fixed_kernel_points,
                                    KP_influence=KP_influence,
                                    aggregation_mode=aggregation_mode)
        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

    def init_KP(self):
        # Create one kernel disposition. Choose the KP distance to center thanks to the KP extent
        if self.use_frame:
            
            kernel_tangents, rotation = load_kernels(manifold=self.manifold,
                                                   radius=self.KP_extent,
                                                   num_kpoints=self.K,
                                                   dimension=self.in_channels,
                                                   use_frame=True,
                                                   temperature=self.temperature,
                                                   sample_ratio=self.sample_ratio,
                                                   use_learnable_rotation=self.use_learnable_rotation)
            self.kernel_points = nn.Parameter(kernel_tangents, requires_grad=False)
            self.rotation = rotation  
    
    @torch.no_grad()
    def get_kernel_pos(self, x, nei, nei_mask, sample, sample_num, transp, radius = None):
        """Get kernel point positions with caching mechanism to avoid repeated computatio"""
        n, d = x.shape
        if radius == None:
            radius = self.KP_extent
            
        device_id = x.device.index if x.device.index is not None else -1
        cache_key = f"{device_id}_{n}_{d}_{transp}_{radius}_{self.deformable}"
        
        if hasattr(self, 'kernel_cache') and cache_key in self.kernel_cache:
            cached_res = self.kernel_cache[cache_key]
            if cached_res.shape[0] == n:  
                return cached_res
            elif n < cached_res.shape[0]:  
                return cached_res[:n]
        
        
        if not transp:
            
            kernel_points = self.kernel_points
            
            if kernel_points.size(-1) != d:
                
                if self.manifold.name == 'Lorentz':
                    
                    kernel_points = kernel_points[:, :d]
            res = self.manifold.expmap0(kernel_points).repeat(n, 1, 1)
        else:
            x_k = x.repeat(1, 1, self.K - 1).view(n, self.K - 1, d) # (n, k-1, d)
            tmp = self.manifold.transp0(x_k, self.kernel_points[1:]) # parallel transport to x
            tmp = self.manifold.expmap(x_k, tmp) # expmap to manifold
            res = torch.concat((tmp, x.view(n, 1, d)), 1) # add fixed kernel (n, k, d)
        
        if self.deformable:
            #print("deformable settings")
            offset = self.offset_conv(x, nei, nei_mask, sample, sample_num) # (n, (d - 1) * k + 1)
            # print(offset)
            offset = self.manifold.split(offset, self.K) # (n, k, d)
            dis = self.manifold.dist0(offset).max()
            offset = self.manifold.logmap0(offset)
            # print(offset)
            offset *= radius / dis
            offset = self.manifold.transp0(res, offset)
            # print(offset)
            res = self.manifold.expmap(res, offset)
        
        
        if not self.training or n > 100:
            if not hasattr(self, 'kernel_cache'):
                self.kernel_cache = {}
            
            if len(self.kernel_cache) > 10:
                
                self.kernel_cache = {}
            self.kernel_cache[cache_key] = res
            
        return res

    def get_nei_kernel_dis(self, x_kernel, x_nei):
        """Calculate the distance between neighbors and kernel points using a memory-optimized version"""
        if x_nei.dim() == 3:
            # corr == 0
            n, nei_num, d = x_nei.shape
            K = x_kernel.shape[1]
            kernel_d = x_kernel.size(-1)

            if kernel_d != d:
                if self.manifold.name == 'Lorentz':
                    if self.layer_idx == 0:
                        
                        spatial_norm_sq = torch.sum(x_nei ** 2, dim=-1, keepdim=True)
                        time_coord = torch.sqrt(1 + spatial_norm_sq)
                        x_nei = torch.cat([time_coord, x_nei], dim=-1)
                    else:
                        
                        x_kernel = x_kernel[:, :, :d]

            
            if x_nei.size(-1) != x_kernel.size(-1):
                raise ValueError(f"Dimension mismatch after adjustment: x_nei: {x_nei.shape}, x_kernel: {x_kernel.shape}")

            
            if n * K * nei_num > 5e6:  
                return self._compute_distance_batched(x_kernel, x_nei, batch_dim=0)
            
            
            # Expand tensors for distance computation
            x_nei_exp = x_nei.unsqueeze(1).expand(n, K, nei_num, x_nei.size(-1))
            x_kernel_exp = x_kernel.unsqueeze(2).expand(n, K, nei_num, x_kernel.size(-1))

            return self.manifold.dist(x_nei_exp, x_kernel_exp)  # (n, K, nei_num)

        elif x_nei.dim() == 4:
            # corr == 1
            n, K, nei_num, d = x_nei.shape
            kernel_d = x_kernel.shape[-1]
            
            
            if kernel_d != d:
                if self.manifold.name == 'Lorentz':
                    if self.layer_idx == 0:
                        
                        spatial_norm_sq = torch.sum(x_nei ** 2, dim=-1, keepdim=True)
                        time_coord = torch.sqrt(1 + spatial_norm_sq)
                        x_nei = torch.cat([time_coord, x_nei], dim=-1)
                    else:
                        
                        x_kernel = x_kernel[:, :, :d]

            
            if x_nei.size(-1) != x_kernel.size(-1):
                raise ValueError(f"Dimension mismatch after adjustment: x_nei: {x_nei.shape}, x_kernel: {x_kernel.shape}")

            
            x_kernel_exp = x_kernel.unsqueeze(2).expand(n, K, nei_num, x_kernel.size(-1))
            return self.manifold.dist(x_nei, x_kernel_exp)

        else:
            raise ValueError(f"x_nei dimension incorrect! Got {x_nei.dim()}-D tensor.")
    
    def _compute_distance_batched(self, x_kernel, x_nei, batch_dim=0):
        """Calculate distances in batches to reduce memory usage"""
        if batch_dim == 0:
            
            n, K, d = x_kernel.shape
            _, nei_num, _ = x_nei.shape
            
            batch_size = max(1, min(32, n // 4))  
            distances = []
            
            for i in range(0, n, batch_size):
                end_i = min(i + batch_size, n)
                x_kernel_batch = x_kernel[i:end_i]
                x_nei_batch = x_nei[i:end_i]
                
                
                x_nei_exp = x_nei_batch.unsqueeze(1).expand(-1, K, nei_num, d)
                x_kernel_exp = x_kernel_batch.unsqueeze(2).expand(-1, K, nei_num, d)
                
                
                dist_batch = self.manifold.dist(x_nei_exp, x_kernel_exp)
                distances.append(dist_batch)
            
            return torch.cat(distances, dim=0)
        else:
            raise NotImplementedError("Only batch_dim=0 is supported")

    def transport_x(self, x, x_nei):
        

        x0_nei = self.manifold.expmap0(self.manifold.transp0back(x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape), self.manifold.logmap(x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape), x_nei) ) )
        x0 = self.manifold.origin(x.shape[-1]).repeat(x.shape[0], 1)
        return x0, x0_nei, x
    
    def apply_kernel_transform(self, x_nei):
        
        if self.K > 32:  
            return self.apply_kernel_transform_memory_efficient(x_nei)
        
        
        res = []
        for k in range(self.K):
            res.append(self.linears[k](x_nei).unsqueeze(1))
        return torch.concat(res, dim = 1)
    
    def apply_kernel_transform_memory_efficient(self, x_nei):
        """Memory-optimized version: Process kernel point transformations one by one to avoid allocating large amounts of memory at once"""
        device = x_nei.device
        n, nei_num, d = x_nei.shape
        
        
        if self.training and n * self.K * nei_num > 1e7:  
            print(f"[INFO] Using gradient checkpointing for kernel transform (n={n}, K={self.K}, nei_num={nei_num})")
            return self._apply_kernel_transform_checkpoint(x_nei)
        
        
        output = torch.zeros(n, self.K, nei_num, self.out_channels, device=device, dtype=x_nei.dtype)
        
        
        batch_size = max(4, min(16, self.K // 4))  
        for k_start in range(0, self.K, batch_size):
            k_end = min(k_start + batch_size, self.K)
            for k in range(k_start, k_end):
                output[:, k, :, :] = self.linears[k](x_nei)
        
        return output
    
    def _apply_kernel_transform_checkpoint(self, x_nei):
        """use gradient checkpointing version"""
        import torch.utils.checkpoint as checkpoint
        
        def compute_kernel_batch(x_nei, k_start, k_end):
            res = []
            for k in range(k_start, k_end):
                res.append(self.linears[k](x_nei).unsqueeze(1))
            return torch.cat(res, dim=1)
        
        num_batches = 4
        batch_size = (self.K + num_batches - 1) // num_batches
        
        results = []
        for i in range(num_batches):
            k_start = i * batch_size
            k_end = min((i + 1) * batch_size, self.K)
            if k_start < k_end:
                if self.training:
                
                    batch_result = checkpoint.checkpoint(
                        compute_kernel_batch, x_nei, k_start, k_end
                    )
                else:
                    
                    batch_result = compute_kernel_batch(x_nei, k_start, k_end)
                results.append(batch_result)
        
        return torch.cat(results, dim=1)
    
    def avg_kernel(self, x_nei_transform, x_nei_kernel_dis):
        """Average the kernel point transformation results with memory optimization suppor"""
        n, K, nei_num, d_out = x_nei_transform.shape
        
        
        if n * nei_num * K > 5e6:  
            return self._avg_kernel_memory_efficient(x_nei_transform, x_nei_kernel_dis)
        
        # original
        x_nei_transform = x_nei_transform.swapaxes(1, 2) # (n, nei_num, k, d')
        x_nei_kernel_dis = x_nei_kernel_dis.swapaxes(1, 2).unsqueeze(3) # (n, nei_num, k)
        return self.manifold.mid_point(x_nei_transform, x_nei_kernel_dis.swapaxes(2, 3)) # (n, nei_num, d')
    
    def _avg_kernel_memory_efficient(self, x_nei_transform, x_nei_kernel_dis):
        """Memory-optimized version of kernel point averaging"""
        n, K, nei_num, d_out = x_nei_transform.shape
        
        batch_size = max(1, min(16, n // 4))
        results = []
        
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            
            batch_transform = x_nei_transform[i:end_i].swapaxes(1, 2)  # (batch, nei_num, k, d')
            batch_dis = x_nei_kernel_dis[i:end_i].swapaxes(1, 2).unsqueeze(3)  # (batch, nei_num, k, 1)
            
            batch_result = self.manifold.mid_point(batch_transform, batch_dis.swapaxes(2, 3))
            results.append(batch_result)
        
        return torch.cat(results, dim=0)


    def sample_nei(self, nei, nei_mask, sample_num):
        new_nei = []
        new_nei_mask = []
        for i in range(len(nei)):
            tot = nei_mask[i].sum()
            if tot > 0:
                new_nei.append(nei[i][torch.randint(0, tot, (sample_num,))])
                new_nei_mask.append(torch.ones((sample_num,), device=nei.device))
            else:
                new_nei.append(torch.zeros((sample_num,), device=nei.device))
                new_nei_mask.append(torch.zeros((sample_num,), device=nei.device))
        return torch.stack(new_nei).type(torch.long), torch.stack(new_nei_mask).type(torch.long)
        
    def forward(self, x, nei, nei_mask, transp = True, sample = False, sample_num = 16):
        # x (n, d) data value
        # nei (n, nei_num) neighbors
        # nei_mask (n, nei_num) 0/1 mask for neighbors
        if sample:
            nei, nei_mask = self.sample_nei(nei, nei_mask, sample_num)
        
        x_nei = gather(x, nei) # (n, nei_num, d)
        if transp:
            x, x_nei, x0 = self.transport_x(x, x_nei)   
        n, nei_num, d = x_nei.shape

        
        x_kernel = self.get_kernel_pos(x, nei, nei_mask, sample, sample_num, transp = not transp) # (n, k, d)

        if self.corr == 0:
            #Use d(xi ominus x, xk)
            x_nei_kernel_dis = self.get_nei_kernel_dis(x_kernel, x_nei) # (n, k, nei_num)
            nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num) # (n, k, nei_num)
            x_nei_kernel_dis = x_nei_kernel_dis * nei_mask
            x_nei_transform = self.apply_kernel_transform(x_nei) # (n, k, nei_num, d')
        elif self.corr == 1:
            #print('corr == 1, Coming here') #Use d(xik, xk)
            x_nei_transform = self.apply_kernel_transform(x_nei) # (n, k, nei_num, d')
            if x_nei.shape[-1] != x_nei_transform.shape[-1]:
                raise ValueError("Don't change dimension in linear transformation step if use corr==1")
            x_nei_kernel_dis = self.get_nei_kernel_dis(x_kernel, x_nei_transform) # (n, k, nei_num)
            nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num) # (n, k, nei_num)
            x_nei_kernel_dis = x_nei_kernel_dis * nei_mask
        else:
            raise NotImplementedError("The specified correlation type is not implemented.")

        x_nei_transform = self.avg_kernel(x_nei_transform, x_nei_kernel_dis).squeeze(2) # (n, nei_num, d')

        if self.nei_agg == 0:
            #######################Uniform neighbor Aggregation#######################
            x_final = self.manifold.mid_point(x_nei_transform) # (n, d')
            #x_final = self.final_linear_act(x_final) # (n, d')
            return x_final
            #######################Uniform neighbor Aggregation#######################
        elif self.nei_agg == 1:
            #######################Attention Neighbor Aggregation#######################
            #print("x_nei_transform.shape:", x_nei_transform.shape)
            attention1 = F.softmax(self.atten1(torch.cat((x_nei_transform, torch.zeros_like(x_nei_transform)),dim=-1)
                                                ).squeeze(-1),dim=-1) #attention(n,nei_num)
            attention2 = F.softmax(self.atten2(torch.cat((x_nei_transform, torch.zeros_like(x_nei_transform)),dim=-1)
                                                ).squeeze(-1),dim=-1) #attention(n,nei_num)
            multihead_attention = ((attention1+attention2)/2).unsqueeze(2)

            #print("multihead_attention.shape:", multihead_attention.swapaxes(1, 2).shape)
            x_final = self.manifold.mid_point(x_nei_transform, multihead_attention.swapaxes(1, 2)).squeeze(1) # (n, d')
            #x_final = self.final_linear_act(x_final) # (n, d')
            #print("x_final.shape:", x_final.shape)
            return x_final
            #######################Attention Neighbor Aggregation#######################
        elif self.nei_agg == 2:
            #######################GIN neighbor Aggregation perspective#######################
            x_final = self.manifold.mid_point(self.MLP_f(x_nei_transform))
            x_final = self.MLP_fi(x_final)
            #print("x_final.shape:", x_final.shape)
            return x_final
            #######################GIN neighbor Aggregation perspective#######################
        else:
            raise NotImplementedError("The specified correlation type is not implemented.")


class KPGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, kernel_size, KP_extent, radius, in_features, out_features, use_bias, dropout, nonlin=None, deformable=False, corr=0, nei_agg=0, use_frame=False, temperature=1.0, sample_ratio=None, use_learnable_rotation=True, layer_idx=0):
        super(KPGraphConvolution, self).__init__()
        print("deformable:", deformable)
        print("nonlin:", nonlin)
        print("corr:", corr)
        print("nei_agg:", nei_agg)
        print("use_frame:", use_frame)
        print("temperature:", temperature)
        print("sample_ratio:", sample_ratio)
        print("use_learnable_rotation:", use_learnable_rotation)
        self.net = KernelPointAggregation(kernel_size, in_features, out_features, KP_extent, radius, manifold, use_bias, dropout, nonlin=nonlin, deformable=deformable, corr=corr, nei_agg=nei_agg, use_frame=use_frame, temperature=temperature, sample_ratio=sample_ratio, use_learnable_rotation=use_learnable_rotation)
        self.net.layer_idx = layer_idx  # 设置层索引

    def forward(self, input):
        x, nei, nei_mask = input
        h = self.net(x, nei, nei_mask)
        output = h, nei, nei_mask
        return output

class KernelPointMidPoint(nn.Module):
    def __init__(self, manifold, kernel_size, KP_extent, radius, in_features, out_features, use_bias, dropout, nonlin=None):
        super(KernelPointMidPoint, self).__init__()
        self.manifold = manifold 
        self.net = KernelPointAggregation(kernel_size, in_features, out_features, KP_extent, radius, manifold, use_bias, dropout, nonlin=nonlin, use_learnable_rotation=False)

    def foward(self, x):
        x0 = self.manifold(x)
        return self.net(x0, x, torch.zeros(x.shape[:-1]).to(x.device()))
