import argparse
import os
import re
import select
import shutil
import sys
import termios
import tty as ttc

import numpy as np

from spume.advection import advection
from spume.diffusion import diffuse_implicit
from spume.projection import pressure_projection
from spume.render import render

# Ansi Escape Sequences
ALT_BUFFER_ON  = "\x1b[?1049h"
ALT_BUFFER_OFF = "\x1b[?1049l"
HIDE_CURSOR    = "\x1b[?25l"
SHOW_CURSOR    = "\x1b[?25h"
MOVE_TOP_LEFT  = "\x1b[H"
ENABLE_MOUSE="\x1b[?1000h\x1b[?1002h\x1b[?1006h"
DISABLE_MOUSE="\x1b[?1002l\x1b[?1006l\x1b[?1000l"

# Spawner shapes — ring 3-4 and ring 6-7 
spawn_offsets = [
    (_dx, _dy)
    for _dx in range(-4, 5)
    for _dy in range(-4, 5)
    if 9 <= _dx * _dx + _dy * _dy <= 16
]

spawn_offsets_large = [
    (_dx, _dy)
    for _dx in range(-6, 7)
    for _dy in range(-6, 7)
    if 36 <= _dx * _dx + _dy * _dy <= 49
]

MOUSE_PAT = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")

def _apply_free_slip(u, v):
    """Free-slip boundaries: tangential copy, normal zero. Deduplicated helper."""
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]
    u[:, 0] = 0
    u[:, -1] = 0
    v[0, :] = 0
    v[-1, :] = 0
    return u, v

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
    parser.add_argument("--gravity", type=float, default=0.003, help="Gravity force")
    parser.add_argument("--dissipate", type=float, default=0.96, help="Dissipation factor for smoke (0.8-0.99)")
    parser.add_argument("--disrupt-radius", type=int, default=4, help="Radius of click impulse in cells (1-20)")
    parser.add_argument("--disrupt-strength", type=float, default=35.0, help="Strength of click impulse (1-70)")
    parser.add_argument("--large-spawner", action="store_true", help="Use a larger smoke spawner (radius 7) instead of the default (radius 4)")
    args = parser.parse_args(argv)
    if not 0.8 <= args.dissipate <= 0.99:
        parser.error("dissipate must be between 0.8 and 0.99")
    if not 1 <= args.disrupt_radius <= 20:
        parser.error("disrupt_radius must be between 1 and 20")
    if not 1 <= args.disrupt_strength <= 70:
        parser.error("disrupt_strength must be between 1 and 70")
    return args.width, args.height, args.dt, args.dx, args.smoke_viscosity, args.viscosity, args.maxiter, args.rtol, args.gravity, args.dissipate, args.disrupt_radius, args.disrupt_strength, args.large_spawner

def update(u, v, p, s, spawn_x, spawn_y, viscosity, smoke_viscosity, dt, dx, maxiter, rtol, gravity, dissipate):
    # Half-step diffusion
    u = diffuse_implicit(u, viscosity, dt / 2, dx, rtol=rtol, maxiter=maxiter)
    v = diffuse_implicit(v, viscosity, dt / 2, dx, rtol=rtol, maxiter=maxiter)
    s = diffuse_implicit(s, smoke_viscosity, dt / 2, dx, rtol=rtol, maxiter=maxiter)

    # Apply forces
    v += gravity * s * dt
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

    u, v = _apply_free_slip(u, v)

    # Project
    u, v, p = pressure_projection(u, v, dx, p, rtol=rtol, maxiter=maxiter)

    u, v = _apply_free_slip(u, v)



    # Dissipate smoke
    s *= dissipate
    return u, v, p, s

def main():
    global spawn_offsets
    grid_width, grid_height, dt, dx, smoke_viscosity, viscosity, maxiter, rtol, gravity, dissipate, disrupt_radius, disrupt_strength, large_spawner  = parse_cli()
    if large_spawner:
        spawn_offsets = spawn_offsets_large

    spawn_radius = 7 if large_spawner else 4
    min_w = 2 * spawn_radius + 3
    min_h = 2 * spawn_radius + 3
    if grid_width < min_w or grid_height < min_h:
        print(f"error: grid {grid_width}x{grid_height} too small for spawner radius {spawn_radius} (need >={min_w}x{min_h})", file=sys.stderr)
        sys.exit(2)
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
            u, v, p, s = update(u, v, p, s, spawn_x, spawn_y, viscosity, smoke_viscosity, dt, dx, maxiter, rtol, gravity, dissipate)
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
                        dis_x = min(max(dis_x, disrupt_radius), grid_height - disrupt_radius - 1)
                        dis_y = min(max(dis_y, disrupt_radius), grid_width - disrupt_radius - 1)

                        yy, xx = np.ogrid[-disrupt_radius:disrupt_radius, -disrupt_radius:disrupt_radius]
                        r = np.hypot(xx, yy) + 1e-5
                        mask = r < disrupt_radius
                        u[dis_x - disrupt_radius:dis_x + disrupt_radius, dis_y - disrupt_radius:dis_y + disrupt_radius] += disrupt_strength * xx / r * mask
                        v[dis_x - disrupt_radius:dis_x + disrupt_radius, dis_y - disrupt_radius:dis_y + disrupt_radius] += disrupt_strength * yy / r * mask

                buf_no_esc = re.sub(rb"\x1b\[[^\x40-\x7E]*[\x40-\x7E]", b"", buf).lower()
                if b"q" in buf_no_esc:
                    break
                spawn_y += 3 * buf_no_esc.count(b'd') - 3 * buf_no_esc.count(b'a')
                spawn_x += 3 * buf_no_esc.count(b'w') - 3 * buf_no_esc.count(b's')
                spawn_y = min(max(spawn_y, spawn_radius), grid_width - spawn_radius - 1)
                spawn_x = min(max(spawn_x, spawn_radius), grid_height - spawn_radius - 1)

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
