
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from advection import advection
from diffusion import diffuse_implicit
from projection import pressure_projection

# Parameters
grid_height = 40
grid_width = 80
dt = 0.1
dx = 1.0
viscosity = 0.001
gravity = 0.01
n_steps = 500

# State
u = np.zeros((grid_height, grid_width))
v = np.zeros((grid_height, grid_width))
p = np.zeros((grid_height, grid_width))

# Visualization setup
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(p, cmap='cividis', origin='lower')
im.set_clim(0, 0.5)
quiver = ax.quiver(
    np.arange(0, grid_width, 2),
    np.arange(0, grid_height, 2),
    u[::2, ::2],
    v[::2, ::2],
    color='black',
    scale=20,
)
plt.colorbar(im, label='Pressure')
ax.set_title('Fluid Simulation')


def update(frame):
    global u, v

    u_old = u.copy()
    v_old = v.copy()

    # Apply gravity 
    v -= gravity

    # Advect
    u = advection(u, u_old, v_old, dt, dx)
    v = advection(v, u_old, v_old, dt, dx)

    # Diffuse
    u = diffuse_implicit(u, viscosity, dt, dx)
    v = diffuse_implicit(v, viscosity, dt, dx)

    # Boundary conditions
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]

    u[:, 0] = 0
    u[:, -1] = 0
    v[0, :] = 0
    v[-1, :] = 0



    # Project
    u, v, p = pressure_projection(u, v, dx)




    # Update visualization
    im.set_data(p)
    quiver.set_UVC(u[::2, ::2], v[::2, ::2])
    ax.set_title(f'Fluid Simulation - Step {frame}')
    return [im, quiver]


ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=30, blit=True)
plt.show()
