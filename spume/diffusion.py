import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def diffuse_implicit(field, viscosity, dt, dx, rtol=1e-5, maxiter=500):
    N1 = field.shape[0]
    N2 = field.shape[1]
    dof = N1 * N2

    def laplacian(flatfield):
        f = flatfield.reshape((N1, N2))
        fp = np.pad(f, 1, mode='edge')
        lap = (fp[2:, 1:-1] + fp[:-2, 1:-1] + fp[1:-1, 2:] + fp[1:-1, :-2] - 4 * f) / dx ** 2
        return flatfield - viscosity * dt * lap.ravel()

    A = LinearOperator(shape=(dof, dof), matvec=laplacian)
    result, info = cg(A, field.flatten(), rtol=rtol, maxiter=maxiter)
    result = result.reshape(N1, N2)
    if info != 0:
        print(f"Warning: diffusion cg failed, info = {info}")
    return result

