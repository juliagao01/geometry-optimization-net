# vicinity_opt

Geometric optimization of a 2D channel-with-obstacle, in Julia, on top of
[FermiSea.jl](https://github.com/jackhfarrell/FermiSea.jl). We parameterize
one obstacle with a polar Fourier series, then search for the shape that
maximizes the linear response coefficient

```
                V_A − V_B
        f_1  =  ─────────       (V_A, V_B = floating-probe potentials)
                    I
```

where `I` is the current injected at the source contact.

## Status & results

The prose below is the **original** design; its early `f_1` values (the `0.385` circle,
`0.972` deformed) were transients and are **retracted** — ignore them.

On this (`main`) branch the trustworthy result is the **obstacle-free vicinity device**:

```
f_1 ≈ 0.020    # mesh-converged, ε-insensitive, α-free (nonlocal spreading resistance)
```
Reproduce: `julia --project=. scripts/fixed_mesh_vicinity.jl`.

The full **obstacle-shape optimization** — a structured **O-grid** giving trustworthy
obstacle numbers (round circle ≈ 0.033, square ≈ 0.094, **optimal deformable oval ≈ 0.144**)
plus nonlinear coefficients (`f_2 ≈ 0.088`, `f_3 ≈ −0.0018`) — lives on the **`nonlinear`
branch** (see its `docs/optimal_shape_explained.md`). See
[`docs/f1_trustworthiness_obstacle_vs_vicinity.md`](docs/f1_trustworthiness_obstacle_vs_vicinity.md)
for the trust audit and `CLAUDE.md` for the code map.

## Why this measurement

For a linear Boltzmann model (which is what `IsotropicFermiHarmonics2D`
solves), `f_1` is the *only* nonzero coefficient in any expansion
`ΔV = f_1 I + f_2 I² + …`. It is the device's vicinity resistance, the
signature distinguishing ballistic / hydrodynamic / ohmic transport in
multiterminal devices (Farrell & Lucas, [arXiv:2605.03030](https://arxiv.org/abs/2605.03030)).

By up-down symmetry of the channel, `f_1 = 0` for any obstacle whose
shape is also y-symmetric (e.g. a centered circle, or a pure-cosine
`r(θ)`). To get a signal you need to break that symmetry — either by
displacing the obstacle (`y_c ≠ 0`) or by including `sin(nθ)` harmonics
in `r(θ)`. We expose both knobs.

## Layout

```
vicinity_opt/
├── Project.toml
├── README.md
├── runs/                        # mesh + result files dropped here at runtime
├── scripts/
│   └── run_optimization.jl      # entry point
└── src/
    ├── VicinityOpt.jl           # top-level module
    ├── geometry.jl              # Fourier obstacle + .geo writer
    ├── mesh.jl                  # Gmsh.jl wrapper (.geo -> .inp)
    ├── simulate.jl              # one FermiSea steady-state solve
    ├── objective.jl             # parameter packing, bounds, validation
    └── optimize.jl              # BlackBoxOptim driver
```

## Parameter vector

`p` has length `2 + 2 M` where `M = cfg.n_modes`:

```
p = [ y_c,       # obstacle vertical offset
      r_0,       # mean radius
      a_1, b_1,  # mode n=1 (translation along x, along y)
      a_2, b_2,  # mode n=2 (ellipticity, tilt)
      ...
      a_M, b_M ]
```

Decoded by `Geometry.unpack_params(p, cfg)` into an `ObstacleParams`.
Default bounds are in `Objective.bounds_for(cfg)`. Default `M = 6`,
giving a 14-dimensional search space.

## Setup

You need Julia 1.10+, Gmsh installed at the system level (the `Gmsh.jl`
Julia wrapper drives a system library), and FermiSea.jl + Trixi.jl
registered with the local Julia.

```bash
cd vicinity_opt
julia --project=. -e 'using Pkg; Pkg.develop(path="../FermiSea.jl"); Pkg.instantiate()'
```

(Replace `../FermiSea.jl` with wherever you cloned it.)

## Running

```bash
julia --project=. scripts/run_optimization.jl
```

This:

1. **Sanity check**: evaluates `f_1` on an off-center circle
   `(y_c=0.1, r_0=0.15)`. Expected: small positive number (probe A sees
   the obstacle, probe B doesn't, so density piles up at A).
2. **Optimization**: 200 evaluations of differential evolution. Writes
   `runs/main/result.jld2`.

Iterate by editing the `cfg`, `sim_cfg`, `opt_cfg` blocks at the top of
the script.

## Physics defaults (and how to change them)

```julia
sim_cfg = SimConfig(
    n_harmonics  = 10,    # angular-harmonic truncation M of the Boltzmann eq
    gamma_mc     = 200,   # momentum-conserving collision rate — large = hydrodynamic
    gamma_mr     = 0.01,  # momentum-relaxing — small = clean (not ohmic)
    I_source     = 1.0,   # driving current; any nonzero is fine since model is linear
    polydeg      = 3,     # DG polynomial degree per cell
)
```

`gamma_mc = 200` with `n_harmonics = 10` is the user's recommendation:
strong momentum-conserving collisions kill harmonics m ≥ 2 fast, so 10
modes is plenty. To explore the ballistic limit, drop `gamma_mc` to 0.1
and consider bumping `n_harmonics` up.

## What the optimization is likely to find

Stokes-flow intuition (hydrodynamic limit `γ_mc → ∞`, no-slip walls)
predicts `f_1` will be largest when:

- the obstacle is lifted close to one wall (`y_c → W/2`);
- the gap is small but not so small that viscous dissipation drowns the
  signal — there is a non-trivial optimum in the gap size;
- higher Fourier modes mostly hurt unless they specifically shape the
  flow into a stronger wall-stress asymmetry.

The Python mock in this directory's parent folder (`mock_pipeline.py`,
`sensitivity_sweep.py`) reproduces this picture qualitatively and is
useful as a fast sanity check before launching the Julia run.

## Reading the result

```julia
using JLD2
@load "runs/main/result.jld2" best_p best_f1 history

using VicinityOpt
cfg = ChannelConfig()
o   = unpack_params(best_p, cfg)
@show o.y_c o.r0
@show o.a_cos
@show o.b_sin
```

To visualize the converged shape, write a `.geo` for it and open in Gmsh:

```julia
write_geo("best.geo", o, cfg)
# in a shell: gmsh best.geo
```

## Likely pain points

- **Mesh quality near tight gaps.** If `r_0 + y_c` approaches `W/2`,
  Gmsh's quad recombination can produce sliver elements that hurt the
  CFL. Tighten `cfg.lc` near the gap or refuse shapes with
  `min_gap < 1.5 * lc`.
- **Discontinuous objective.** Two parameter vectors that differ by 1%
  may produce meshes with different element counts and slightly
  different `f_1`. Use a derivative-free optimizer (we default to
  DE/CMA-ES), not a gradient method.
- **Wall-clock per evaluation.** A single steady-state solve at
  `polydeg=3, n_harmonics=10` is the dominant cost. Cache evaluations
  by parameter hash if you'll be sweeping.

## Extensions worth trying

- Release `x_c` as a free parameter — currently fixed at channel center.
- Add the outer-wall sine series `y_top(x) = W/2 + Σ s_n sin(nπx/L)`
  on top of the rectangular channel. This is the second parameterization
  in the original prompt and gives the optimizer another way to break
  symmetry.
- Two obstacles: parameterize each independently and let the optimizer
  decide whether to merge / split them.
- Magnetic field: `MagneticFieldSource(equations, B)` is already in
  FermiSea. Adding `B ≠ 0` introduces a true Hall contribution and
  changes what "asymmetry" means.
