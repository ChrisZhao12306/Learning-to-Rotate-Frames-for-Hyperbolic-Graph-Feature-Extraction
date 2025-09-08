from geoopt import manifolds
import torch
from torch.nn import functional as F
from torch.distributions import Normal, Independent
from numbers import Number
from torch.distributions.utils import _standard_normal, broadcast_all

from manifolds.base import Manifold


class LorentzWrappedNormal(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        raise NotImplementedError

    @property
    def scale(self):
        return F.softplus(self._scale) if self.softplus else self._scale

    def __init__(self, loc, scale, manifold, dim = 1, c = 0, validate_args=None, softplus=False):
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold
        self.dim = dim
        self.c = c
        self.device = loc.device
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.dim])
        super(LorentzWrappedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        tmp = list(shape)
        tmp[-1] -= 1
        shape = torch.Size(tmp)
        v = self.scale[:-1] * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        v = torch.cat([torch.zeros_like(v).narrow(-1, 0, 1), v], dim=-1)
        if torch.allclose(self.manifold.origin(self.dim), self.loc):
            z = self.manifold.expmap0(v)
        else:
            u = self.manifold.transp0(y = self.loc, u = v)
            z = self.manifold.expmap(x = self.loc, u = u)
        return z