import torch
from torch.nn import functional as F
from torch.distributions import Normal, Independent
from numbers import Number
from torch.distributions.utils import _standard_normal, broadcast_all


class WrappedNormal(torch.distributions.Distribution):

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
        # self.manifold.assert_check_point_on_manifold(self.loc)
        self.device = loc.device
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.dim])
        super(WrappedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        v = self.scale * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        # self.manifold.assert_check_vector_on_tangent(self.manifold.zero, v)
        v = v / self.manifold._lambda_x(x=torch.zeros(1, self.dim).to(self.loc.device), c=self.c)
        if torch.allclose(torch.zeros(self.dim).to(self.loc.device), self.loc):
            z = self.manifold.expmap0(v, self.c)
        else:
            u = self.manifold.ptransp(x=torch.zeros(1, self.dim).to(self.loc.device), y=self.loc, u=v, c=self.c)
            z = self.manifold.expmap(u = self.loc, p = u, c = self.c)
        return z