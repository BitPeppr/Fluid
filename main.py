import shutil
import sys

import numpy as np

from advection import advection
from diffusion import diffuse_implicit
from projection import pressure_projection
from render import render

# Ansi Escape Sequences
ALT_BUFFER_ON  = "\x1b[?1049h"
ALT_BUFFER_OFF = "\x1b[?1049l"
HIDE_CURSOR    = "\x1b[?25l"
SHOW_CURSOR    = "\x1b[?25h"
MOVE_TOP_LEFT  = "\x1b[H"


# Parameters
grid_height = 80
grid_width = 200
grid_width, grid_height = shutil.get_terminal_size((grid_width, grid_height))
dt = 0.5
dx = 1.0
smoke_viscosity = 0.00001
viscosity = 0.01

# State
u = np.zeros((grid_height, grid_width))
v = np.zeros((grid_height, grid_width))
p = np.zeros((grid_height - 2, grid_width - 2))
s = np.zeros((grid_height, grid_width))


def update():
    global u, v, p, s

    
    # Half-step diffusion
    u = diffuse_implicit(u, viscosity, dt / 2, dx)
    v = diffuse_implicit(v, viscosity, dt / 2, dx)
    s = diffuse_implicit(s, smoke_viscosity, dt / 2, dx)

    # Apply forces 
    v += 0.003 * s * dt
    s[19:28, 47:50] += 0.5

    u_old = u.copy()
    v_old = v.copy()

    # Advect
    u = advection(u, u_old, v_old, dt, dx)
    v = advection(v, u_old, v_old, dt, dx)
    s = advection(s, u_old, v_old, dt, dx)

    # Finish half-step diffusion
    u = diffuse_implicit(u, viscosity, dt / 2 , dx)
    v = diffuse_implicit(v, viscosity, dt / 2, dx)
    s = diffuse_implicit(s, smoke_viscosity, dt / 2, dx)

    # Boundary conditions (free-slip)
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]

    u[:, 0] = 0; u[:, -1] = 0
    v[0, :] = 0; v[-1, :] = 0


    # Project
    u, v, p = pressure_projection(u, v, dx, p)


    # Boundary conditions (free-slip)
    u[:, 0] = 0
    u[:, -1] = 0
    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]

    v[0, :] = 0
    v[-1, :] = 0
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]



    # Dissipate smoke
    s *= 0.98
    return s

def main():
    sys.stdout.write(ALT_BUFFER_ON + HIDE_CURSOR)
    sys.stdout.flush()

    try:
        while True:
            s = update()
            rendered = render(s)
            sys.stdout.write(MOVE_TOP_LEFT + rendered)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(ALT_BUFFER_OFF + SHOW_CURSOR)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
