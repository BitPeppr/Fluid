import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def pressure_projection(u, v, dx, p0, rtol=1e-6, maxiter=200):
    N1 = u.shape[0]
    N2 = u.shape[1]

    div =  (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dx)
    shape_interior = (N1-2, N2-2)
    dof = (N1-2) * (N2-2)

    def poisson_laplacian(flat_p):
        p = flat_p.reshape(shape_interior)
        padp = np.pad(p, 1, mode='edge')
        lap = (4*p - padp[2:, 1:-1] - padp[:-2, 1:-1] - padp[1:-1, 2:] - padp[1:-1, :-2]) / (dx**2)
        flat = lap.flatten()
        return flat

    rhs = -div.ravel()
    rhs -= np.mean(rhs)

    if p0 is not None:
        if p0.shape == (N1, N2):
            p0 = p0[1:-1, 1:-1]
        x0 = p0.ravel().copy()
        x0 -= np.mean(x0)
    else:
        x0 = None

    p, info = cg(LinearOperator((dof, dof), matvec=poisson_laplacian), rhs, x0=x0, rtol=rtol, maxiter=maxiter)
    p = p.reshape(shape_interior)

    p = np.pad(p, 1, mode='edge')
    p -= np.mean(p)

    u[1:-1, 1:-1] -= (p[1:-1, 1:-1] - p[1:-1, :-2]) / dx
    v[1:-1, 1:-1] -= (p[1:-1, 1:-1] - p[:-2, 1:-1]) / dx

    return u, v, p[1:-1, 1:-1]
