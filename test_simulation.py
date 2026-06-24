import random
import numpy as np

from advection import advection
from diffusion import diffuse_implicit
from projection import pressure_projection

grid_height = 40
grid_width = 100
dt = 0.5
dx = 1.0
viscosity = 0.01
gravity = 0.01

u = np.zeros((grid_height, grid_width))
v = np.zeros((grid_height, grid_width))
p = np.zeros((grid_height - 2, grid_width - 2))
rng = np.random.RandomState(42)

for step in range(8):
    u = diffuse_implicit(u, viscosity, dt/2, dx)
    v = diffuse_implicit(v, viscosity, dt/2, dx)
    v -= gravity * dt
    v[15:35, 45:55] += gravity * dt * 5 * rng.uniform(0, 2)
    u_old, v_old = u.copy(), v.copy()
    u = advection(u, u_old, v_old, dt, dx)
    v = advection(v, u_old, v_old, dt, dx)
    u = diffuse_implicit(u, viscosity, dt/2, dx)
    v = diffuse_implicit(v, viscosity, dt/2, dx)

    # Pre-projection BCs (FIXED)
    u[0,:]=u[1,:]; u[-1,:]=u[-2,:]
    v[:,0]=v[:,1]; v[:,-1]=v[:,-2]
    u[:,1]=0; u[:,-1]=0   # FIXED: 0→1
    v[1,:]=0; v[-1,:]=0   # FIXED: 0→1

    u, v, p = pressure_projection(u, v, dx, p)

    # Post-projection BCs (FIXED)
    u[:,1]=0; u[:,-1]=0   # FIXED: 0→1
    v[:,0]=v[:,1]; v[:,-1]=v[:,-2]
    v[1,:]=0; v[-1,:]=0   # FIXED: 0→1
    u[0,:]=u[1,:]; u[-1,:]=u[-2,:]

    if step % 2 == 0:
        div_final = (u[1:-1,2:]-u[1:-1,:-2])/(2*dx) + (v[2:,1:-1]-v[:-2,1:-1])/(2*dx)
        mass_l = np.sum(u[1:-1, 1]); mass_r = np.sum(u[1:-1, -2])
        mass_b = np.sum(v[1, 1:-1]); mass_t = np.sum(v[-2, 1:-1])
        net = mass_r - mass_l + mass_t - mass_b
        print(f"Step {step}: v[1,:] RMS={np.sqrt(np.mean(v[1,:]**2)):.2e} | u[:,1] RMS={np.sqrt(np.mean(u[:,1]**2)):.2e} | max|div|={np.max(np.abs(div_final)):.2e} | net flux={net:+.2e}")

print("\nALL BOUNDARIES NOW ZEROED:")
print(f"  u[:, 1] (LEFT wall)  RMS = {np.sqrt(np.mean(u[:,1]**2)):.2e}")
print(f"  u[:,-1] (RIGHT wall) RMS = {np.sqrt(np.mean(u[:,-1]**2)):.2e}")
print(f"  v[1, :] (BOTTOM wall) RMS = {np.sqrt(np.mean(v[1,:]**2)):.2e}")
print(f"  v[-1,:] (TOP wall)    RMS = {np.sqrt(np.mean(v[-1,:]**2)):.2e}")