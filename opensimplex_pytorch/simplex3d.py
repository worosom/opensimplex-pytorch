import torch
from .pcg import pcg3d
from .misc import fill, dot, step, to_float


# 3d simplex noise
# https://www.shadertoy.com/view/XsX3zB
class Simplex3D:
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        # skew constants for 3d simplex functions
        self.F3 = fill((3,), torch.tensor(1/3), device)
        self.G3 = fill((3,), torch.tensor(1/6), device)

    @torch.jit.script
    def get_s_x(p, F3, G3):
        _p = torch.einsum('in,n->i', p, F3)
        s = torch.floor(p + _p[..., None])
        x = p - s + torch.einsum('in,n->i', s, G3)[..., None]
        return s, x


    @torch.jit.script
    def get_i(x, edge):
        _val = x - torch.roll(x, (-1), dims=-1)
        e = step(edge, _val)
        e[:, 2] = torch.min(e[:, 2], 3. - dot(e, torch.tensor(1.)))
        i1 = e * (1 - torch.roll(e, (1), dims=-1))
        i2 = 1 - torch.roll(e, (1), dims=-1) * (1 - e)
        return i1, i2


    @torch.jit.script
    def get_x3(x, i1, i2, G3):
        x1 = x - i1 + G3
        x2 = x - i2 + 2. * G3
        x3 = x - 1. + 3. * G3
        return x1, x2, x3


    @torch.jit.script
    def get_surflets(x, x1, x2, x3, s, i1, i2, zero):
        w = torch.dstack((
            dot(x, x),
            dot(x1, x1),
            dot(x2, x2),
            dot(x3, x3)
        ))

        # w fades from 0.6 at the center of the surflet to 0.0 at the margin
        w = torch.maximum(0.6 - w, zero)

        # calculate surflet components
        d = torch.dstack((
            dot(to_float(pcg3d(s)), x),
            dot(to_float(pcg3d(s + i1)), x1),
            dot(to_float(pcg3d(s + i2)), x2),
            dot(to_float(pcg3d(s + 1.)), x3)
        ))

        # multiply d by w^4
        w = torch.pow(w, 4)
        d = torch.multiply(d, w)
        return w, d
    
    def __call__(self, p):
        p = p.type(torch.float32)
        # 1. find current tetrahedron T and it's four vertices
        # s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices
        # x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices

        # calculate s and x
        s, x = self.get_s_x(p, self.F3, self.G3)

        # calculate i1 and i2
        edge = torch.zeros((x.shape[0], 3), dtype=torch.float32, device=self.device)
        i1, i2 = self.get_i(x, edge)

        # x1, x2, x3
        x1, x2, x3 = self.get_x3(x, i1, i2, self.G3)

        # 2. find four surflets and store them in d

        # calculate surflet weights
        zero = torch.tensor(0.0, device=self.device)
        w, d = self.get_surflets(x, x1, x2, x3, s, i1, i2, zero)

        weight = fill((d.shape[0], 4), torch.tensor(200.0), device=self.device)
        # 3. return the sum of the four surflets
        value = dot(d, weight)

        return value
