import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from advection import advection
from diffusion import diffuse_implicit
from projection import pressure_projection

# Parameters
grid_height = 40
grid_width = 100
dt = 0.5
dx = 1.0
viscosity = 0.01
gravity = 0.01
n_steps = 20

# State
u = np.zeros((grid_height, grid_width))
v = np.zeros((grid_height, grid_width))
p = np.zeros((grid_height - 2, grid_width - 2))

# Visualization setup
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(v, cmap='cividis', origin='lower')
im.set_clim(-0.5, 0.5)
quiver = ax.quiver(
    np.arange(0, grid_width, 2),
    np.arange(0, grid_height, 2),
    u[::2, ::2],
    v[::2, ::2],
    color='black',
    scale=20,
)
ax.set_title('Fluid Simulation')


def update(frame):
    global u, v, p

    
    # Half-step diffusion
    u = diffuse_implicit(u, viscosity, dt / 2, dx)
    v = diffuse_implicit(v, viscosity, dt / 2, dx)

    # Apply gravity 
    v -= gravity * dt
    v[15:35, 45:55] += gravity * dt * 5  * random.uniform(0, 2)  # Add a burst of upward velocity in the center

    u_old = u.copy()
    v_old = v.copy()

    # Advect
    u = advection(u, u_old, v_old, dt, dx)
    v = advection(v, u_old, v_old, dt, dx)

    # Finish half-step diffusion
    u = diffuse_implicit(u, viscosity, dt / 2 , dx)
    v = diffuse_implicit(v, viscosity, dt / 2, dx)

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



    # Update visualization
    im.set_data(v)
    quiver.set_UVC(u[::2, ::2], v[::2, ::2])
    ax.set_title(f'Fluid Simulation - Step {frame}')
    return [im, quiver]


ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=30, blit=True)
plt.show()
