import numpy as np
import torch


def to_float32(x):
    return (x.type(torch.float64) / (torch.iinfo(x.dtype).max - torch.iinfo(x.dtype).min)).type(torch.float32)


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = torch.rand(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*torch.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*torch.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = torch.sqrt(z)
    Vx, Vy, Vz = V = torch.tensor((
        torch.sin(phi) * r,
        torch.cos(phi) * r,
        torch.sqrt(2.0 - z)
        ))

    st = torch.sin(theta)
    ct = torch.cos(theta)

    R = torch.tensor(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (torch.outer(V, V) - torch.eye(3)).matmul(R)
    return M

