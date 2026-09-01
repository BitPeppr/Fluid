import argparse
import os
import re
import select
import shutil
import sys
import termios
import tty as ttc

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
ENABLE_MOUSE="\x1b[?1000h\x1b[?1002h\x1b[?1006h"
DISABLE_MOUSE="\x1b[?1002l\x1b[?1006l\x1b[?1000l"

spawn_offsets = [
        (0, 0),
        (1, 0), (0, 1), (-1, 0), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2),
        (3, 0), (0, 3)
        ]

MOUSE_PAT = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")

def parse_cli(argv=None):
    default_width, default_height = shutil.get_terminal_size((200, 80))
    parser = argparse.ArgumentParser(description="Terminal-based 2D fluid simulation")
    parser.add_argument("--width", type=int, default=default_width, help="Grid width")
    parser.add_argument("--height", type=int, default=default_height, help="Grid height")
    parser.add_argument("--dt", type=float, default=0.5, help="Time step")
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing")
    parser.add_argument("--smoke-viscosity", type=float, default=0.00001, help="Smoke viscosity")
    parser.add_argument("--viscosity", type=float, default=0.01, help="Fluid viscosity")
    parser.add_argument("--maxiter", type=int, default=500, help="Maximum iterations for pressure solver")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for pressure solver")
    args = parser.parse_args(argv)
    return args.width, args.height, args.dt, args.dx, args.smoke_viscosity, args.viscosity, args.maxiter, args.rtol

def update(u, v, p, s, spawn_x, spawn_y, viscosity, smoke_viscosity, dt, dx, maxiter, rtol):
    # Half-step diffusion
    u = diffuse_implicit(u, viscosity, dt / 2, dx, rtol=rtol, maxiter=maxiter)
    v = diffuse_implicit(v, viscosity, dt / 2, dx, rtol=rtol, maxiter=maxiter)
    s = diffuse_implicit(s, smoke_viscosity, dt / 2, dx, rtol=rtol, maxiter=maxiter)

    # Apply forces
    v += 0.003 * s * dt
    for ax, ay in spawn_offsets:
        s[spawn_x + ax, spawn_y + ay] += 1

    u_old = u.copy()
    v_old = v.copy()

    # Advect
    u = advection(u, u_old, v_old, dt, dx)
    v = advection(v, u_old, v_old, dt, dx)
    s = advection(s, u_old, v_old, dt, dx)

    # Finish half-step diffusion
    u = diffuse_implicit(u, viscosity, dt / 2 , dx, rtol=rtol, maxiter=maxiter)
    v = diffuse_implicit(v, viscosity, dt / 2, dx, rtol=rtol, maxiter=maxiter)
    s = diffuse_implicit(s, smoke_viscosity, dt / 2, dx, rtol=rtol, maxiter=maxiter)

    # Boundary conditions (free-slip)
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]

    u[:, 0] = 0; u[:, -1] = 0
    v[0, :] = 0; v[-1, :] = 0


    # Project
    u, v, p = pressure_projection(u, v, dx, p, rtol=rtol, maxiter=maxiter)


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
    return u, v, p, s

def main():
    grid_width, grid_height, dt, dx, smoke_viscosity, viscosity, maxiter, rtol = parse_cli()
    spawn_x = 10
    spawn_y = 10

    u = np.zeros((grid_height, grid_width))
    v = np.zeros((grid_height, grid_width))
    p = np.zeros((grid_height - 2, grid_width - 2))
    s = np.zeros((grid_height, grid_width))

    original_settings = termios.tcgetattr(sys.stdin)
    ttc.setcbreak(sys.stdin.fileno())
    sys.stdout.write(ALT_BUFFER_ON + HIDE_CURSOR + ENABLE_MOUSE)
    sys.stdout.flush()
    buf = b""
    should_exit = False

    try:
        while True:
            u, v, p, s = update(u, v, p, s, spawn_x, spawn_y, viscosity, smoke_viscosity, dt, dx, maxiter, rtol)
            rendered = render(s)
            sys.stdout.write(MOVE_TOP_LEFT + rendered)
            sys.stdout.flush()

            if should_exit:
                break

            if select.select([sys.stdin], [], [], 0)[0]:
                buf += os.read(sys.stdin.fileno(), 64)
                while True:
                    m = MOUSE_PAT.search(buf)
                    if not m:
                        break
                    cb, col, row = int(m.group(1)), int(m.group(2)) - 1, int(m.group(3)) - 1
                    buf = buf[:m.start()] + buf[m.end():]
                    if cb in (0, 32):
                        dis_x = grid_height - 1 - row
                        dis_y = col
                        dis_x = min(max(dis_x, 3), grid_height - 4)
                        dis_y = min(max(dis_y, 5), grid_width - 6)

                        yy, xx = np.ogrid[-3:3, -5:5]
                        r = np.hypot(xx, yy) + 1e-5
                        stren = 24
                        u[dis_x - 3:dis_x + 3, dis_y - 5:dis_y + 5] += stren * xx / r * (r < 5)
                        v[dis_x - 3:dis_x + 3, dis_y - 5:dis_y + 5] += stren * yy / r * (r < 5)

                if b"q" in buf:
                    break
                spawn_y += 3 * buf.count(b'd') - 3 * buf.count(b'a')
                spawn_x += 3 * buf.count(b'w') - 3 * buf.count(b's')
                spawn_y = min(max(spawn_y, 5), grid_width - 6)
                spawn_x = min(max(spawn_x, 3), grid_height - 4)

                if b"\x1b[<" in buf:
                    idx = buf.rfind(b"\x1b")
                    if idx != -1 and len(buf) - idx < 12:
                        buf = buf[idx:]
                    else:
                        buf = b""
                elif buf.endswith(b"\x1b") or buf.endswith(b"\x1b["):
                    pass
                else:
                    buf = b""

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
        sys.stdout.write(DISABLE_MOUSE + ALT_BUFFER_OFF + SHOW_CURSOR)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
