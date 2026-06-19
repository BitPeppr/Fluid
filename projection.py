import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def pressure_projection(u, v, dx):
    N1 = u.shape[0]
    N2 = u.shape[1]

    div = np.zeros((N1, N2))
    div[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dx)

    def poisson_laplacian(flat_p):
        p = flat_p.reshape((N1, N2))
        lap = np.zeros_like(p)
        lap[1:-1, 1:-1] = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4*p[1:-1, 1:-1]) / (dx**2)
        return lap.flatten()
    p, info = cg(LinearOperator((N1*N2, N1*N2), matvec=poisson_laplacian), div.flatten(), rtol=1e-5, maxiter=500)
    p = p.reshape((N1, N2))
    p -= np.mean(p)
    
    p[0,:] = p[1,:]
    p[-1,:] = p[-2,:]
    p[:,0] = p[:,1]
    p[:,-1] = p[:,-2]

    u[1:-1, 1:-1] -= (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dx)
    v[1:-1, 1:-1] -= (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dx)


    return u, v, p
