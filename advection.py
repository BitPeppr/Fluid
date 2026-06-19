import numpy as np


def advection(field, u, v, dt, dx):
    N1 = field.shape[0]
    N2 = field.shape[1]
    result = np.zeros_like(field)

    for i in range(1, N1-1):
        for j in range(1, N2-1):
            x_back = i - v[i, j] * dt / dx
            y_back = j - u[i, j] * dt / dx

            x_back = np.clip(x_back, 0, N1 - 1)
            y_back = np.clip(y_back, 0, N2 - 1)

            i0 = int(x_back)
            j0 = int(y_back)
            i1 = min(i0+1, N1-1)
            j1 = min(j0+1, N2-1)

            fx = x_back - i0
            fy = y_back - j0

            result[i, j] = (field[i0, j0] * (1-fx) * (1-fy) +
                             field[i1, j0] * fx * (1-fy) +
                                field[i0, j1] * (1-fx) * fy +
                                field[i1, j1] * fx * fy)
    return result
