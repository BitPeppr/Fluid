import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def diffuse_implicit(field, viscosity, dt, dx):
    N1 = field.shape[0]
    N2 = field.shape[1]
    dof = N1 * N2

    def laplacian(flatfield):
        f = flatfield.reshape((N1, N2))
        lap = np.zeros_like(f)
        lap[1:-1, 1:-1] = (f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1]) / (dx ** 2)
        return (flatfield - viscosity * dt * lap.flatten())

    A = LinearOperator(shape=(dof, dof), matvec=laplacian)
    result = cg(A, field.flatten(), rtol=1e-5, maxiter=500)[0].reshape(N1, N2)
    return result

