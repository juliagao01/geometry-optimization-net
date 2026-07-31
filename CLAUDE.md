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

`src/VicinityOpt.jl` includes and re-exports six submodules. The dependency
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

### Parameter vector

`p = [y_c, r0, a_1, b_1, ..., a_M, b_M]`, length `2 + 2M` where
`M = cfg.n_modes`. Decode with `unpack_params(p, cfg)`. The obstacle is
`r(θ) = r0 + Σ (a_n cos nθ + b_n sin nθ)` centered at `(cfg.x_c, y_c)`;
`x_c` is fixed at channel center. By up-down symmetry, `f_1 = 0` unless the
shape breaks y-symmetry (offset `y_c ≠ 0`, or nonzero `b_sin`).

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
