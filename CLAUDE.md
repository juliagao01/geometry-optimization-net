# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`VicinityOpt` (dir name `geometry-optimization-net`) is a Julia package that
optimizes the shape of a 2D channel-with-obstacle to maximize the vicinity-
resistance coefficient `f_1 = (V_A − V_B) / I`, where `V_A`, `V_B` are
self-consistent floating-probe potentials and `I` is the injected source
current. Physics is solved by the external **FermiSea.jl** (linear Boltzmann
moment / `IsotropicFermiHarmonics2D` model) on a Trixi DG mesh. See `README.md`
for the physics motivation and the arXiv reference.

The pipeline per objective evaluation: pack parameters → build a polar Fourier
obstacle → write a Gmsh `.geo` → mesh to Abaqus `.inp` → FermiSea steady-state
solve → read `f_1`. A derivative-free optimizer loops this.

## ⚠️ Retracted results — read before trusting any f_1 number

**All earlier `f_1` results are RETRACTED as transients**, including the
`0.385` circle and `0.972` deformed-shape numbers cited in the README, scripts,
and commit history. They came from a **misconfigured `SteadyStateCallback`** —
the solve was read off before reaching steady state. Do not cite, warm-start
from, or compare against these numbers.

The callback **must** be configured `SteadyStateCallback(; abstol=1e-4,
reltol=0.0)`. Any new `f_1` is only trustworthy once produced with this
configuration and the stabilizing physics params below.

### Canonical stabilizing params (from advisor)

Use these in `SimConfig` unless you have a specific reason not to:

```julia
n_harmonics = 4
gamma_mc    = 300.0
gamma_mr    = 0.05
polydeg     = 1
cfl         = 0.5
```

**Always launch Julia with `--threads=auto`**, e.g.
`julia --threads=auto --project=. scripts/run_optimization.jl`.

These supersede the values shown in `README.md`, existing scripts, and the
`SimConfig` defaults, which reflect the retracted tuning.

## Setup and commands

FermiSea.jl is a **local path dependency** (`Manifest.toml` points at
`../cu/FermiSea.jl`) — it must be present and developed, and Gmsh must be
installed at the system level (the `Gmsh.jl` wrapper drives a system library).

```bash
# First-time install (adjust FermiSea path to wherever it lives)
julia --project=. -e 'using Pkg; Pkg.develop(path="../cu/FermiSea.jl"); Pkg.instantiate()'

# Main entry point (sanity check + optimization run)
julia --project=. scripts/run_optimization.jl

# Any of the exploratory scripts follow the same pattern
julia --project=. scripts/<name>.jl
```

There is **no test suite and no lint config**. Verification is done by running
scripts and inspecting `f_1` values / `runs/*/result.jld2`. Every script begins
with `Pkg.activate(joinpath(@__DIR__, ".."))`, so run from anywhere.

## Architecture

`src/VicinityOpt.jl` includes and re-exports seven submodules. The dependency
order is important — each `include` depends on the ones above it:

1. **`geometry.jl` (`Geometry`)** — `ChannelConfig`, `ObstacleParams`,
   `unpack_params`/`pack_params`, `sample_obstacle`, cheap `validate` (rejects
   geometries before meshing), and `write_geo` (emits the Gmsh `.geo`).
2. **`mesh.jl` (`Mesh`)** — `geo_to_inp`, a headless Gmsh wrapper (quasi-
   structured quad algorithm + recombine → Abaqus `.inp`).
3. **`solver_interface.jl` (`SolverInterface`)** — the `AbstractSolverConfig` +
   `evaluate(config, inp_path) -> NamedTuple` contract that keeps everything
   else solver-agnostic. Add a new physics backend by implementing an
   `evaluate` method for a new config type.
4. **`simulate.jl` (`Simulate`)** — `SimConfig` and the FermiSea solve. This is
   the dominant cost. Boundary-condition symbols (`:contact_source`,
   `:contact_drain`, `:probe_A`, `:probe_B`, `:walls`) **must match the
   `Physical Curve` names emitted in `write_geo`** — changing one requires
   changing the other.
5. **`objective.jl` (`Objective`)** — `bounds_for(cfg)`, `EvalState` (mutable,
   holds the eval counter + full `history`), `evaluate_once`, and
   `make_objective` which returns a `p -> -f_1` closure (optimizers minimize;
   invalid/failed geometries return a positive `penalty` instead of NaN).
6. **`optimize.jl` (`Optimize`)** — `OptConfig` + `run_optimization`, driving
   **Optim.jl `ParticleSwarm`**, saving `best_p`, `best_f1`, `history` to
   `runs/*/result.jld2`.
7. **`fixed_mesh.jl` (`FixedMesh`)** — the current line of work (see below):
   density/Brinkman obstacle on a fixed mesh + the exact assembled-operator
   solver and adjoint.

### Parameter vector

`p = [y_c, r0, a_1, b_1, ..., a_M, b_M]`, length `2 + 2M` where
`M = cfg.n_modes`. Decode with `unpack_params(p, cfg)`. The obstacle is
`r(θ) = r0 + Σ (a_n cos nθ + b_n sin nθ)` centered at `(cfg.x_c, y_c)`;
`x_c` is fixed at channel center. By up-down symmetry, `f_1 = 0` unless the
shape breaks y-symmetry (offset `y_c ≠ 0`, or nonzero `b_sin`).

## Fixed-mesh density method (`FixedMesh`, the current line of work)

Instead of remeshing an obstacle boundary per candidate, mesh the empty channel
**once** and represent the obstacle as a per-cell density `ρ(x) ∈ [0,1]`. A cell
is made "solid" by **Brinkman penalization** — a source term that damps the
momentum harmonics by `α·ρ(x)` (leaving the density mode a0 alone), so high-ρ
cells block current like a wall. This makes the objective smooth in ρ, enabling
gradient/topology optimization. Key API (`src/fixed_mesh.jl`):

- `DensityField`, `BrinkmanSource`, `paint_blob!` — the ρ grid + coupling.
- `write_channel_geo_symmetric` — mirror-symmetric structured mesh (mesh it with
  `geo_to_inp(...; structured=true)`, which skips the Algorithm-11 override that
  would otherwise destroy the transfinite grid).
- `write_point_geo` — **narrow point-contact** vicinity geometry (3×3 transfinite
  blocks, y-symmetric).
- `run_f1_fixed_steady` — matrix-free GMRES steady solve.
- `assemble_fixed_operator` → `FixedOperator`, then `f1_exact(op, ρ; alpha, eps)`
  (exact reg-LU) and `f1_adjoint_grad(op, ρ; alpha, eps)` (full gradient from 2
  solves; verified vs finite-diff to 1e-6). **This is the canonical solver** —
  scripts predate it and inline the same assembly.

### Hard-won facts (do not re-derive)

- **Time-stepping is hopeless** here (stiff + slow-diffusive); one solve was
  ~1000 s. Use the direct solvers.
- **The steady operator is singular** — the undamped density mode a0 is a null
  vector, so `A u = -b` is non-unique and `f_1` was ill-defined. Fixed with
  **Tikhonov (min-norm) regularization** (`+εI`, ε≈1e-10); `f_1` is stable as
  ε→0. This is the root cause of the project's historic `f_1` instability.
- **Full-edge contacts give f_1 ≈ 0 for ANY obstacle** (uniform 1-D flow, no
  vicinity signal — verified: reg-LU `f_1 ∝ ε → 0`). **Narrow point contacts**
  create the spreading 2-D flow that carries a real signal.
- **The obstacle must be solid** (α ≈ 2000, not 100 — at α=100 current flows
  straight through). High-α solves are stiff; GMRES stalls, so use `f1_exact`.
- **Optimum shape** = a large circle (radius ≈ channel width) with **sin-harmonic
  dimples**: a plain big circle must be near-centered (size-vs-offset constraint)
  → `f_1=0` by symmetry, so the **dimples** (e.g. `b2` sin-2θ) break symmetry and
  carry the signal. Result at α=2000: `f_1 ≈ 3×10²`, ε-convergent to machine
  precision, mesh-robust to ~5%. Saved to `runs/fixed_mesh/result_circle.jld2`.
- **`f_1` magnitude scales ∝ α (no hard-wall limit)**: solidity sweep gives f_1 =
  37/312/3046/15305 at α = 200/2000/2e4/1e5. So the absolute magnitude is
  α-dependent (a Brinkman property for this boundary observable) — optimize and
  compare shapes at a *fixed* α; "f_1≈312" means α=2000. f_1 is nearly independent
  of γ_mc (ballistic↔hydrodynamic, ~2%) and of contact width for wc≲0.3 (~7%),
  collapsing to 0 only at the full-edge (wc→W) limit.

### Environment / running

Finer meshes get **OOM/SIGTERM-killed** here; run heavy Julia with
`julia --threads=2 --gcthreads=1 --heap-size-hint=6G`. Julia block-buffers
stdout to files — `flush(stdout)` in scripts. Gmsh GUIs pop up via
`gmsh.fltk.run()`; **do not mesh a jagged obstacle-as-hole** (`generate(2)`
hangs in edge recovery) — show CAD geometry or the fixed mesh + a boundary
overlay instead. Visualize the optimized shape with
`RESULT=result_circle.jld2 julia --project=. scripts/show_gmsh_grid.jl` (fixed
mesh grid + obstacle boundary) or `.../show_gmsh_boundary.jl` (boundary in
perimeter); `scripts/make_plot.jl` builds an HTML figure via FermiSea's
`save_cartesian`.

## Watch out for (code vs. README drift)

The README describes an earlier design; the code has since changed. Trust the
code:

- **Default `n_modes = 0`** in `ChannelConfig` (README says 6). Most current
  runs are the 2-parameter circle case (`[y_c, r0]`); Fourier runs set
  `n_modes` explicitly.
- **Optimizer**: `Optimize.run_optimization` uses Optim.jl `ParticleSwarm`
  (README says BlackBoxOptim/DE). `scripts/fourier_opt*.jl` instead drive
  `CMAEvolutionStrategy.minimize` directly rather than going through `Optimize`.
- **Two solve paths in `simulate.jl`**: `run_f1` (time-steps to steady state
  with `ROCK4` + `SteadyStateCallback`; this is what `evaluate` calls and the
  only path that works — but only once the callback is configured as in the
  retraction section above) and `run_f1_steady` (direct matrix-free
  GMRES/BiCGStab Krylov solve). **`run_f1_steady` FAILS unpreconditioned — do
  not use it** until preconditioning is added.
- Physics params in `SimConfig` defaults and across scripts reflect the
  retracted, misconfigured tuning. Use the canonical stabilizing params from
  the retraction section, not the in-file defaults.

## Layout notes

- `scripts/` — the actual working surface: `run_optimization.jl` is the entry
  point; the rest are one-off convergence checks, sweeps, and optimizer
  experiments (`mesh_convergence*`, `*_sweep`, `fourier_opt*`, `steady_*`,
  `grid_search`). `mock_pipeline.py` / `sensitivity_sweep.py` are a Python
  surrogate (no FermiSea) for validating the framework and sanity-checking
  hydrodynamic intuition cheaply.
- `runs/` — per-experiment output dirs; each holds generated `shape_N.{geo,inp}`
  meshes and a `result.jld2` (`@load` gives `best_p`, `best_f1`, `history`).
- `blocky/` — a drafted alternative **blocky Cartesian** obstacle
  representation (vs. the polar Fourier one). Python side is tested; the Julia
  side is **untested**. See `blocky/CHECKLIST.md` for status before touching it.
- `_old/` — a superseded flat (non-modular) version of the code. Reference only.
- Top-level `*.inp` / `*.geo` / `*_p4est_ready.inp` files are standalone
  meshes/geometries used during solver debugging.
