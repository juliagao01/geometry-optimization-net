"""
Geometry prototype for vicinity-resistance optimization in FermiSea.jl.

Parameterization (matches what we'll write in geometry.jl):

    Channel:   rectangle [0, L_x] x [-W/2, W/2]
               Source contact on x = 0 (length W).
               Drain  contact on x = L_x (length W).
               Floating probe A:  top wall around x = x_probe, length L_probe.
               Floating probe B:  bottom wall around x = x_probe, length L_probe.
               Remaining boundary: MaxwellWall.

    Obstacle:  ONE closed curve, polar parameterization centered at (x_c, y_c):

                   r(theta) = r_0 + sum_{n=1..M} a_n cos(n theta) + b_n sin(n theta)

               Default: pure cosine series r(theta) = r_0 + sum a_n cos(n theta),
               which is what the prompt specifies. We include sin terms behind a
               flag because they let you break the chirality of the obstacle and
               may matter for f_1 once the channel itself isn't y-symmetric.

The response we want to maximize:

    f_1 = (n_A - n_B) / I        in the linear-response limit I -> 0,

where n_A, n_B are the contact potentials of the two floating probes (the
a_0 modes assembled by FloatingProbeBC). Because the Boltzmann model is
linear, f_1 is independent of I, so we just pick one small I.

This file does no simulation. It just builds the boundary curves and writes
a Gmsh .geo file, so we can sanity-check the parameterization visually
before plugging it into FermiSea.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import os


# ---------------------------------------------------------------------------
# Channel + obstacle parameterization
# ---------------------------------------------------------------------------

def obstacle_polar(theta: np.ndarray,
                   r0: float,
                   a_cos: np.ndarray,
                   b_sin: np.ndarray | None = None,
                   x_c: float = 0.0,
                   y_c: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Polar parameterization of one closed obstacle.

        r(theta) = r0 + sum_n a_cos[n-1] * cos(n*theta) + b_sin[n-1] * sin(n*theta)

    Returns (x, y) sampled on a closed loop (last point == first point).
    """
    n_modes = len(a_cos)
    if b_sin is None:
        b_sin = np.zeros(n_modes)
    assert len(b_sin) == n_modes, "a_cos and b_sin must have the same length"

    r = np.full_like(theta, r0, dtype=float)
    for n in range(1, n_modes + 1):
        r += a_cos[n - 1] * np.cos(n * theta)
        r += b_sin[n - 1] * np.sin(n * theta)

    # Enforce positivity. If a user hands us coefficients that produce r<=0
    # we clip; the optimizer will learn to avoid that region via the penalty.
    r = np.maximum(r, 1e-3)

    x = x_c + r * np.cos(theta)
    y = y_c + r * np.sin(theta)
    return x, y


def channel_corners(L_x: float, W: float) -> tuple[np.ndarray, np.ndarray]:
    """Outer rectangular channel, counter-clockwise starting at bottom-left."""
    x = np.array([0.0, L_x, L_x, 0.0])
    y = np.array([-W / 2, -W / 2, W / 2, W / 2])
    return x, y


# ---------------------------------------------------------------------------
# Validation: does this shape make sense?
# ---------------------------------------------------------------------------

def obstacle_is_valid(x_obs, y_obs, L_x, W,
                      margin: float = 0.02) -> tuple[bool, str]:
    """Cheap geometry checks to bail out before meshing."""
    if np.any(x_obs < margin) or np.any(x_obs > L_x - margin):
        return False, "obstacle extends past channel in x"
    if np.any(y_obs < -W / 2 + margin) or np.any(y_obs > W / 2 - margin):
        return False, "obstacle extends past channel in y"
    # No self-intersection check here. For r(theta) with bounded coefficients
    # relative to r0, self-intersection is impossible iff |r'(theta)| stays
    # finite and r > 0. We already clip r>0. Self-intersection of (x,y) on a
    # star-shaped curve about (x_c,y_c) cannot happen.
    return True, "ok"


def plot_geometry(L_x: float, W: float,
                  x_c: float, y_c: float,
                  r0: float, a_cos: np.ndarray, b_sin: np.ndarray | None,
                  x_probe: float, L_probe: float,
                  ax=None, title: str = ""):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Outer channel
    cx, cy = channel_corners(L_x, W)
    ax.fill(cx, cy, color="#f1f3f5", edgecolor="black", linewidth=1.2)

    # Source/drain contacts (vertical edges, colored)
    ax.plot([0, 0], [-W / 2, W / 2], color="#1971c2", linewidth=3,
            label="source (current I)")
    ax.plot([L_x, L_x], [-W / 2, W / 2], color="#c92a2a", linewidth=3,
            label="drain (current -I)")

    # Floating probes
    ax.plot([x_probe - L_probe / 2, x_probe + L_probe / 2],
            [W / 2, W / 2], color="#2f9e44", linewidth=3, label="probe A (top)")
    ax.plot([x_probe - L_probe / 2, x_probe + L_probe / 2],
            [-W / 2, -W / 2], color="#9c36b5", linewidth=3, label="probe B (bot)")

    # Obstacle
    theta = np.linspace(0, 2 * np.pi, 400)
    ox, oy = obstacle_polar(theta, r0, a_cos, b_sin, x_c, y_c)
    ax.fill(ox, oy, color="#fcc419", edgecolor="black", linewidth=1.0,
            alpha=0.85, label="obstacle")

    ax.set_aspect("equal")
    ax.set_xlim(-0.05 * L_x, 1.05 * L_x)
    ax.set_ylim(-0.7 * W, 0.7 * W)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    return ax


# ---------------------------------------------------------------------------
# Emit Gmsh .geo for the Julia side
# ---------------------------------------------------------------------------

def write_geo(path: str,
              L_x: float, W: float,
              x_c: float, y_c: float,
              r0: float, a_cos: np.ndarray, b_sin: np.ndarray | None,
              x_probe: float, L_probe: float,
              n_obstacle_points: int = 120,
              lc: float = 0.04) -> None:
    """
    Emit a .geo file that defines the outer channel, the four contact
    segments on its boundary, and one closed obstacle as a hole.
    Boundary curves get Physical Curve names that match what we'll pass
    to FermiSea as boundary_symbols.
    """
    # --- outer channel boundary as 8 vertices to carve out the probes ---
    # Bottom wall goes left -> right with the bottom probe in the middle.
    # Top    wall goes right -> left with the top probe in the middle.
    xL, xR = 0.0, L_x
    yT, yB = +W / 2, -W / 2
    xPL = x_probe - L_probe / 2
    xPR = x_probe + L_probe / 2

    outer = [
        (xL, yB),    # 1 source bottom
        (xPL, yB),   # 2 start of bottom probe
        (xPR, yB),   # 3 end of bottom probe
        (xR, yB),    # 4 drain bottom
        (xR, yT),    # 5 drain top
        (xPR, yT),   # 6 end of top probe
        (xPL, yT),   # 7 start of top probe
        (xL, yT),    # 8 source top
    ]

    # --- obstacle samples ---
    theta = np.linspace(0, 2 * np.pi, n_obstacle_points, endpoint=False)
    ox, oy = obstacle_polar(theta, r0, a_cos, b_sin, x_c, y_c)

    lines = []
    lines.append('SetFactory("OpenCASCADE");')
    lines.append(f"lc = {lc};")
    lines.append("")
    # Outer points
    for i, (x, y) in enumerate(outer, start=1):
        lines.append(f"Point({i}) = {{{x:.8f}, {y:.8f}, 0, lc}};")
    # Outer lines, ccw
    n_outer = len(outer)
    for i in range(1, n_outer + 1):
        j = i + 1 if i < n_outer else 1
        lines.append(f"Line({i}) = {{{i}, {j}}};")
    # Outer loop
    line_ids_outer = list(range(1, n_outer + 1))
    lines.append(f"Curve Loop(1) = {{{', '.join(str(k) for k in line_ids_outer)}}};")

    # Obstacle points and spline
    p_start = n_outer + 1
    for i, (x, y) in enumerate(zip(ox, oy)):
        lines.append(f"Point({p_start + i}) = {{{x:.8f}, {y:.8f}, 0, lc}};")
    spline_pts = list(range(p_start, p_start + len(ox))) + [p_start]
    lines.append(f"Spline({n_outer + 1}) = {{{', '.join(str(k) for k in spline_pts)}}};")
    lines.append(f"Curve Loop(2) = {{{n_outer + 1}}};")

    # Surface with hole
    lines.append("Plane Surface(1) = {1, 2};")

    # Physical groups. Match these names in the Julia simulation.
    lines.append('Physical Surface("domain") = {1};')
    # contact_source: line 8 (xL,yT) -> (xL,yB)? Actually our outer goes
    # 1->2->...->8->1. So edge from point 8 to point 1 is the leftmost vertical
    # segment, which is the source. That's line 8.
    lines.append('Physical Curve("contact_source") = {8};')
    # drain: line 4 (point 4 -> point 5)
    lines.append('Physical Curve("contact_drain")  = {4};')
    # probe_A (top, line 6: point 6 -> point 7)
    lines.append('Physical Curve("probe_A")        = {6};')
    # probe_B (bottom, line 2: point 2 -> point 3)
    lines.append('Physical Curve("probe_B")        = {2};')
    # walls: every other outer line (1,3,5,7) and the obstacle (n_outer+1)
    walls = [1, 3, 5, 7, n_outer + 1]
    lines.append(f'Physical Curve("walls")          = {{{", ".join(str(k) for k in walls)}}};')

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Demo: render a few example shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    L_x, W = 1.0, 0.6
    x_probe, L_probe = 0.5, 0.15
    x_c, y_c = 0.5, 0.0

    examples = [
        ("circle  (r0=0.18, no harmonics)",
         dict(r0=0.18, a_cos=np.zeros(8), b_sin=None)),
        ("ellipse-ish (r0=0.18, a2=0.05)",
         dict(r0=0.18, a_cos=np.array([0, 0.05, 0, 0, 0, 0, 0, 0]), b_sin=None)),
        ("triangle-ish (a3=0.06)",
         dict(r0=0.18, a_cos=np.array([0, 0, 0.06, 0, 0, 0, 0, 0]), b_sin=None)),
        ("teardrop (a1=0.08)",
         dict(r0=0.18, a_cos=np.array([0.08, 0, 0, 0, 0, 0, 0, 0]), b_sin=None)),
        ("flat-top (a2=-0.05, a4=-0.02)",
         dict(r0=0.20, a_cos=np.array([0, -0.05, 0, -0.02, 0, 0, 0, 0]), b_sin=None)),
        ("up-down asym (b1=0.06)",
         dict(r0=0.18, a_cos=np.zeros(8),
              b_sin=np.array([0.06, 0, 0, 0, 0, 0, 0, 0]))),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for ax, (name, kw) in zip(axes.ravel(), examples):
        plot_geometry(L_x, W, x_c, y_c,
                      r0=kw["r0"], a_cos=kw["a_cos"], b_sin=kw["b_sin"],
                      x_probe=x_probe, L_probe=L_probe,
                      ax=ax, title=name)
    plt.tight_layout()
    out_png = "/home/claude/geometry_examples.png"
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"wrote {out_png}")

    # Also emit a sample .geo file
    write_geo("/home/claude/sample.geo", L_x, W, x_c, y_c,
              r0=0.18, a_cos=np.array([0, 0.05, 0, 0, 0, 0, 0, 0]),
              b_sin=np.array([0.04, 0, 0, 0, 0, 0, 0, 0]),
              x_probe=x_probe, L_probe=L_probe)
    print("wrote /home/claude/sample.geo")
