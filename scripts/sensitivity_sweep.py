"""
Sensitivity sweep over plain circular obstacles.

Plots:
    1. f_1 vs y_c at fixed r0 (the user's "asymmetry knob")
    2. f_1 vs r0 at the optimum y_c (the "size knob")
    3. 2D heatmap f_1(y_c, r0)

Same surrogate physics as mock_pipeline.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from mock_pipeline import ChannelConfig, surrogate_f1, validate

cfg = ChannelConfig()
M = cfg.n_modes

def f1_circle(y_c, r0):
    """Pure circle: a_n = b_n = 0."""
    p = [y_c, r0] + [0.0] * (2 * M)
    ok, _ = validate(p, cfg)
    if not ok:
        return np.nan
    return surrogate_f1(p, cfg)


# ---- 1D sweep over y_c at fixed r0 ----
y_grid = np.linspace(-0.13, 0.13, 50)
r0_values = [0.08, 0.12, 0.16, 0.20]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
for r0 in r0_values:
    f1s = [f1_circle(y, r0) for y in y_grid]
    ax.plot(y_grid, f1s, label=f"r0 = {r0:.2f}", linewidth=1.6)
ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("y_c (obstacle offset)")
ax.set_ylabel("f_1 (vicinity response, surrogate)")
ax.set_title("Antisymmetric in y_c, blows up near the wall")
ax.legend()
ax.grid(alpha=0.3)

# ---- 1D sweep over r0 at y_c = 0.10 ----
ax = axes[1]
r_grid = np.linspace(0.05, 0.27, 60)
for y_c in [0.0, 0.06, 0.10, 0.14]:
    f1s = [f1_circle(y_c, r) for r in r_grid]
    ax.plot(r_grid, f1s, label=f"y_c = {y_c:.2f}", linewidth=1.6)
ax.set_xlabel("r0 (obstacle radius)")
ax.set_ylabel("f_1 (surrogate)")
ax.set_title("Larger r0 squeezes the gap, raising f_1")
ax.legend()
ax.grid(alpha=0.3)

# ---- 2D heatmap ----
ax = axes[2]
Y, R = np.meshgrid(y_grid, r_grid, indexing="xy")
F = np.array([[f1_circle(y, r) for y in y_grid] for r in r_grid])
im = ax.pcolormesh(Y, R, F, cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(F)), vmax=np.nanmax(np.abs(F)),
                   shading="auto")
plt.colorbar(im, ax=ax, label="f_1 (surrogate)")
ax.set_xlabel("y_c")
ax.set_ylabel("r0")
ax.set_title("f_1(y_c, r0) -- white regions: invalid geometry")

# Annotate the analytical maximum: pushed against top wall, r0 large.
# Mark the surrogate maximum on the heatmap.
imax = np.unravel_index(np.nanargmax(F), F.shape)
ax.plot(y_grid[imax[1]], r_grid[imax[0]], "k*", markersize=14,
        label=f"max = {F[imax]:.3f}")
ax.legend()

plt.tight_layout()
plt.savefig("/home/claude/sensitivity_sweep.png", dpi=110, bbox_inches="tight")
plt.close()

# Print
imax_y, imax_r = y_grid[imax[1]], r_grid[imax[0]]
print(f"surrogate max within circular family:  f_1 = {F[imax]:.4f}")
print(f"  at y_c = {imax_y:.3f}, r0 = {imax_r:.3f}")
print(f"  min gap to top wall = {cfg.W/2 - imax_y - imax_r:.4f}")
print()
print("wrote /home/claude/sensitivity_sweep.png")
