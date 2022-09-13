import torch


# http://www.jcgt.org/published/0009/03/02/
# https://www.shadertoy.com/view/XlGcRh
def pcg3d(v):
    v = (v * 1664525 + 1013904223).type(torch.int32)

    v[:, 0] += v[:, 1] * v[:, 2]
    v[:, 1] += v[:, 2] * v[:, 0]
    v[:, 2] += v[:, 0] * v[:, 1]

    v ^= v >> 16

    v[:, 0] += v[:, 1] * v[:, 2]
    v[:, 1] += v[:, 2] * v[:, 0]
    v[:, 2] += v[:, 0] * v[:, 1]

    return v


def pcg4d(v):
    v = (v * 1664525 + 1013904223).type(torch.int32)

    v[:, 0] += v[:, 1]*v[:, 3]
    v[:, 1] += v[:, 2]*v[:, 0]
    v[:, 2] += v[:, 0]*v[:, 1]
    v[:, 3] += v[:, 1]*v[:, 2]

    v ^= v >> 16

    v.x += v[:, 1]*v[:, 3]
    v.y += v[:, 2]*v[:, 0]
    v.z += v[:, 0]*v[:, 1]
    v.w += v[:, 1]*v[:, 2]

    return v

