import torch
from .pcg import pcg3d
from .misc import fill, dot, step, to_float


# skew constants for 3d simplex functions
F3 = fill((3,), 1/3)
G3 = fill((3,), 1/6)


# 3d simplex noise
# https://www.shadertoy.com/view/XsX3zB
def simplex3d(p):
    p = p.type(torch.float32)
    # 1. find current tetrahedron T and it's four vertices
    # s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices
    # x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices

    # calculate s and x
    _p = torch.einsum('in,n->i', p, F3)
    s = torch.floor(p + _p[..., None])
    x = p - s + torch.einsum('in,n->i', s, G3)[..., None]

    # calculate i1 and i2
    edge = torch.zeros((x.shape[0], 3), dtype=torch.float32)
    _val = x - torch.roll(x, (-1), dims=-1)
    e = step(edge, _val)
    e[:, 2] = torch.min(e[:, 2], 3. - dot(e, 1.))
    i1 = e * (1 - torch.roll(e, (1), dims=-1))
    i2 = 1 - torch.roll(e, (1), dims=-1) * (1 - e)

    # x1, x2, x3
    x1 = x - i1 + G3
    x2 = x - i2 + 2. * G3
    x3 = x - 1. + 3. * G3

    # 2. find four surflets and store them in d

    # calculate surflet weights
    w = torch.dstack((
        dot(x, x),
        dot(x1, x1),
        dot(x2, x2),
        dot(x3, x3)
    ))

    # w fades from 0.6 at the center of the surflet to 0.0 at the margin
    w = torch.maximum(0.6 - w, torch.zeros((1)))

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

    # 3. return the sum of the four surflets
    value = dot(d, fill((d.shape[0], 4), 52.0))

    return value
