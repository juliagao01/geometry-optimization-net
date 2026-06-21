"""
Python mock of the Julia VicinityOpt pipeline.

Replaces the (expensive, here-unavailable) FermiSea.jl simulator with a
physically-motivated surrogate that captures the expected hydrodynamic
linear-response behavior:

    f_1(shape) = + (lift y_c toward a wall) * (proximity-to-wall amplification)
                 - (viscous dissipation cost from high Fourier modes)

We use this only to validate the *framework*: parameter packing, geometry
validation, bounds, optimizer wrapper. Once you run the real FermiSea on a
real machine, swap surrogate_f1() for the Julia simulation and the rest of
the pipeline stays identical.

Outputs:
    - mock_history.png       : optimization trajectory
    - mock_best_shape.png    : the best shape found
    - mock_best_params.txt   : the parameters of the best shape
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import differential_evolution
import json

# ----- mirror the Julia ChannelConfig --------------------------------------

@dataclass
class ChannelConfig:
    L_x: float = 1.0
    W: float = 0.6
    x_probe: float = 0.5
    L_probe: float = 0.15
    x_c: float = 0.5
    n_modes: int = 6
    margin: float = 0.02

# ----- parameter packing (mirror geometry.jl) ------------------------------

def unpack(p, cfg: ChannelConfig):
    M = cfg.n_modes
    assert len(p) == 2 + 2 * M, f"got len(p)={len(p)}, expected {2+2*M}"
    y_c, r0 = p[0], p[1]
    a = np.zeros(M); b = np.zeros(M)
    for n in range(1, M + 1):
        a[n - 1] = p[2 + 2 * n - 2]
        b[n - 1] = p[2 + 2 * n - 1]
    return y_c, r0, a, b

def sample_obstacle(theta, y_c, r0, a, b, cfg: ChannelConfig):
    r = np.full_like(theta, r0, dtype=float)
    for n in range(1, len(a) + 1):
        r += a[n - 1] * np.cos(n * theta) + b[n - 1] * np.sin(n * theta)
    r = np.maximum(r, 1e-3)
    x = cfg.x_c + r * np.cos(theta)
    y = y_c + r * np.sin(theta)
    return x, y, r

def validate(p, cfg: ChannelConfig):
    y_c, r0, a, b = unpack(p, cfg)
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    x, y, r = sample_obstacle(theta, y_c, r0, a, b, cfg)
    if x.min() < cfg.margin or x.max() > cfg.L_x - cfg.margin: return False, "x oob"
    if y.min() < -cfg.W/2 + cfg.margin or y.max() > cfg.W/2 - cfg.margin: return False, "y oob"
    if r.min() < 0.02: return False, "r collapsed"
    dr = np.zeros_like(theta)
    for n in range(1, len(a) + 1):
        dr += -n * a[n - 1] * np.sin(n * theta) + n * b[n - 1] * np.cos(n * theta)
    if np.max(np.abs(dr) / r) > 5.0: return False, "oscillating"
    return True, "ok"

def bounds_for(cfg: ChannelConfig):
    M = cfg.n_modes
    bnd = [(-cfg.W/4, cfg.W/4),
           (0.05, min(cfg.W/2, cfg.L_x/2) - cfg.margin)]
    for n in range(1, M + 1):
        amp = 0.05 / np.sqrt(n)
        bnd.append((-amp, amp))
        bnd.append((-amp, amp))
    return bnd

# ----- surrogate physics ---------------------------------------------------
#
# f_1 here is a stand-in for what FermiSea would return. The functional form
# is chosen to reproduce the qualitative behavior expected in the hydrodynamic
# limit (gamma_mc large, gamma_mr small): vicinity voltage is roughly
#
#   V ~ (eta * I / W^2) * F(obstacle geometry)
#
# where eta is viscosity and F captures the asymmetry. We pick a tractable F
# that rewards (1) breaking up-down symmetry, (2) sitting close to a wall,
# and penalizes (3) rough/wiggly boundaries (viscous dissipation).

def surrogate_f1(p, cfg: ChannelConfig) -> float:
    y_c, r0, a, b = unpack(p, cfg)
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    x, y, r = sample_obstacle(theta, y_c, r0, a, b, cfg)

    # 1) Effective lever arm: signed area-weighted y-centroid.
    #    For a star-shaped curve about (x_c, y_c), the y-centroid of the
    #    enclosed region is y_c plus a small Fourier-coefficient correction.
    #    Use it as the up-down-asymmetry handle.
    y_centroid = y.mean()  # ~ y_c plus higher-order corrections

    # 2) Gap to the closest wall (top, bottom, source, drain).
    gap_top    = cfg.W/2 - y.max()
    gap_bottom = y.min() - (-cfg.W/2)
    gap_left   = x.min()
    gap_right  = cfg.L_x - x.max()
    min_gap    = min(gap_top, gap_bottom, gap_left, gap_right)
    if min_gap < cfg.margin:
        return float("nan")

    # 3) "Hydrodynamic" prefactor. Vicinity voltage ~ 1/gap^alpha in the
    #    Stokes-flow lubrication limit. Use alpha = 1 (squeeze flow).
    proximity = 1.0 / max(min_gap, 1e-3)

    # 4) Symmetry-breaking signed lever; sign of f_1 should follow sign of y_c.
    asym = y_centroid

    # 5) Roughness penalty from high Fourier coefficients (viscous loss).
    M = cfg.n_modes
    weights = np.arange(1, M + 1, dtype=float) ** 2   # n^2 penalty: sharp shapes cost more
    roughness = (weights * (a ** 2 + b ** 2)).sum()

    # 6) Assemble. Numerical scales chosen so f_1 ~ O(0.1) at the optimum.
    f1 = 1.5 * asym * proximity - 8.0 * roughness
    return float(f1)

# ----- optimization wrapper ------------------------------------------------

class EvalLog:
    def __init__(self):
        self.records = []
        self.best_so_far = []
        self.best_f1 = -np.inf

    def __call__(self, p, cfg):
        ok, why = validate(p, cfg)
        if not ok:
            # soft penalty proportional to "how invalid": we just return a
            # constant; the optimizer will pull back to feasible region.
            val = 1.0
            self.records.append(dict(status="invalid", reason=why, f1=None))
        else:
            f1 = surrogate_f1(p, cfg)
            if np.isnan(f1):
                val = 1.0
                self.records.append(dict(status="bad_geom", reason="gap<margin", f1=None))
            else:
                val = -f1   # minimize -f_1
                self.records.append(dict(status="ok", f1=f1, p=list(p)))
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_p = list(p)
        self.best_so_far.append(self.best_f1)
        return val


def main():
    cfg = ChannelConfig()
    bnd = bounds_for(cfg)
    log = EvalLog()

    print("=== Python mock pipeline ===")
    print(f"dimensions = {len(bnd)}")

    # Stage 1: sanity check
    print("\nSanity: off-center circle (y_c=0.10, r0=0.15)")
    p_san = [0.10, 0.15] + [0.0] * (2 * cfg.n_modes)
    ok, why = validate(p_san, cfg)
    print(f"  validate: {ok} ({why})")
    print(f"  surrogate f_1 = {surrogate_f1(p_san, cfg):.4f}")

    p_centered = [0.0, 0.15] + [0.0] * (2 * cfg.n_modes)
    print("Sanity: centered circle (y_c=0, r0=0.15)")
    print(f"  surrogate f_1 = {surrogate_f1(p_centered, cfg):.4f}   <- expect ~0 by symmetry")

    # Stage 2: real optimization (differential evolution -- same family as
    # the Julia BlackBoxOptim default).
    print("\nOptimizing ...")
    res = differential_evolution(
        lambda p: log(p, cfg),
        bnd, seed=42, popsize=16, maxiter=40, polish=True, tol=1e-4,
        updating="deferred", workers=1, init="sobol",
    )
    print(f"\nbest f_1 = {log.best_f1:.4f}")
    print(f"best p   = {[round(x, 4) for x in log.best_p]}")
    y_c, r0, a, b = unpack(log.best_p, cfg)
    print(f"  y_c = {y_c:.4f}  r0 = {r0:.4f}")
    print(f"  a_cos = {[round(x, 4) for x in a]}")
    print(f"  b_sin = {[round(x, 4) for x in b]}")

    # ------ plots ------
    # trajectory
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    ax.plot(log.best_so_far)
    ax.set_xlabel("evaluation"); ax.set_ylabel("best f_1 so far")
    ax.set_title("optimization trajectory (Python mock)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/mock_history.png", dpi=110)
    plt.close()

    # best shape
    from geometry_prototype import plot_geometry
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_geometry(cfg.L_x, cfg.W, cfg.x_c, y_c,
                  r0, a, b,
                  cfg.x_probe, cfg.L_probe,
                  ax=ax, title=f"mock optimum: f_1 = {log.best_f1:.4f}")
    plt.tight_layout()
    plt.savefig("/home/claude/mock_best_shape.png", dpi=110)
    plt.close()

    # also save best params for re-use
    with open("/home/claude/mock_best_params.json", "w") as fh:
        json.dump(dict(y_c=y_c, r0=r0, a_cos=a.tolist(), b_sin=b.tolist(),
                       f1=log.best_f1, cfg=cfg.__dict__), fh, indent=2)

    print("\nwrote /home/claude/mock_history.png")
    print("wrote /home/claude/mock_best_shape.png")
    print("wrote /home/claude/mock_best_params.json")
    return log


if __name__ == "__main__":
    main()
