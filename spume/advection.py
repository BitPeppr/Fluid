import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def advection_single(field, u, v, dt, dx):
    N1, N2 = field.shape
    result = field.copy()

    ii, jj = np.meshgrid(np.arange(1, N1-1), np.arange(1, N2-1), indexing='ij')
    i_back = ii - v[1:N1-1, 1:N2-1] * dt / dx
    j_back = jj - u[1:N1-1, 1:N2-1] * dt / dx
    i_back = np.clip(i_back, 0, N1 - 1)
    j_back = np.clip(j_back, 0, N2 - 1)

    i0 = i_back.astype(int)
    j0 = j_back.astype(int)
    i1 = np.minimum(i0 + 1, N1 - 1)
    j1 = np.minimum(j0 + 1, N2 - 1)

    fx = i_back - i0
    fy = j_back - j0

    result[1:-1, 1:-1] = (
            field[i0, j0] * (1-fx) * (1-fy) +
            field[i1, j0] * fx * (1-fy) +
            field[i0, j1] * (1-fx) * fy +
            field[i1, j1] * fx * fy
            )
    return result

def advection(field, u, v, dt, dx):
    fwd = advection_single(field, u, v, dt, dx)
    bwd = advection_single(fwd, -u, -v, dt, dx)
    corrected = fwd + 0.5 * (field - bwd)

    local_min = minimum_filter(field, size=3, mode='nearest')
    local_max = maximum_filter(field, size=3, mode='nearest')
    result = np.clip(corrected, local_min, local_max)

    result[0, :] = field[0, :]
    result[-1, :] = field[-1, :]
    result[:, 0] = field[:, 0]
    result[:, -1] = field[:, -1]

    return result


