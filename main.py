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

# Interactive elements
spawn_x = 10
spawn_y = 10

# Parameters
grid_height = 80
grid_width = 200
grid_width, grid_height = shutil.get_terminal_size((grid_width, grid_height))
dt = 0.5
dx = 1.0
smoke_viscosity = 0.00001
viscosity = 0.01

spawn_offsets = [
        (0, 0),
        (1, 0), (0, 1), (-1, 0), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
        (2, 0), (0, 2), (-2, 0), (0, -2),
        (3, 0), (0, 3)
        ]

# State
u = np.zeros((grid_height, grid_width))
v = np.zeros((grid_height, grid_width))
p = np.zeros((grid_height - 2, grid_width - 2))
s = np.zeros((grid_height, grid_width))
def update():
    global u, v, p, s, spawn_x, spawn_y

    
    # Half-step diffusion
    u = diffuse_implicit(u, viscosity, dt / 2, dx)
    v = diffuse_implicit(v, viscosity, dt / 2, dx)
    s = diffuse_implicit(s, smoke_viscosity, dt / 2, dx)

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

# pre-compile mouse pattern (SGR 1006: ESC[<Cb;Col;RowM/m)
MOUSE_PAT = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")

def main():
    global spawn_x, spawn_y
    original_settings = termios.tcgetattr(sys.stdin)
    ttc.setcbreak(sys.stdin.fileno())
    sys.stdout.write(ALT_BUFFER_ON + HIDE_CURSOR + ENABLE_MOUSE)
    sys.stdout.flush()
    buf = b""
    should_exit = False

    try:
        while True:
            s = update()
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
