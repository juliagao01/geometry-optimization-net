# scripts/

Drivers for the two obstacle-optimization approaches. The **canonical solver +
adjoint now live in `src/fixed_mesh.jl`** (`assemble_fixed_operator`, `f1_exact`,
`f1_adjoint_grad`, `write_point_geo`); most scripts below predate that and inline
the same assembly. Run heavy Julia with the memory-lean flags (finer meshes get
OOM-killed here):

```bash
julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/<name>.jl
```

## Canonical / current

| script | purpose |
|---|---|
| `fixed_mesh_circle_opt.jl` | **the result**: optimize a large circle's Fourier dimples → f₁≈3e2 |
| `fixed_mesh_point_opt.jl` | free-form (per-cell) adjoint topology optimization (regularized) |
| `fixed_mesh_tightconv.jl` | mesh convergence of a saved optimized shape |
| `test_fixed_api.jl` | end-to-end check of the promoted `src/` API |
| `show_gmsh_grid.jl` | Gmsh GUI: fixed mesh grid + optimized obstacle boundary overlay |
| `show_gmsh_boundary.jl` | Gmsh GUI: obstacle boundary as a hole in the channel perimeter |
| `make_plot.jl` | HTML figure (current field + obstacle) via FermiSea `save_cartesian` |
| `pointgeo.jl` | (legacy shim) `write_point_geo`; now also in `src/fixed_mesh.jl` |
| `run_optimization.jl` | original boundary-fitted Fourier optimizer (pre-fixed-mesh) |

Reproduce/redisplay the headline result:
```bash
julia ... scripts/fixed_mesh_circle_opt.jl        # writes runs/fixed_mesh/result_circle.jld2
RESULT=result_circle.jld2 julia --project=. scripts/show_gmsh_grid.jl
```

## Diagnostic trail (kept for the record; each answered one question)

- `fixed_mesh_validate.jl`, `fixed_mesh_diag.jl`, `fixed_mesh_bench.jl` — early
  validation + why time-stepping is hopeless (~1000 s/solve).
- `fixed_mesh_steady.jl` — GMRES steady-solver tuning.
- `fixed_mesh_sym.jl`, `fixed_mesh_symcheck.jl` — symmetric mesh; the
  antisymmetrization dead-end.
- `fixed_mesh_debug.jl`, `fixed_mesh_reg.jl`, `fixed_mesh_alpha.jl` — localized the
  **singular density-mode** and the Tikhonov (min-norm) fix.
- `fixed_mesh_regime.jl`, `fixed_mesh_converge.jl` — γ_mc and n_harmonics
  convergence; showed the signal needs a **solid** obstacle (α≈2000).
- `fixed_mesh_point.jl`, `fixed_mesh_point_lu.jl`, `fixed_mesh_point_conv.jl`,
  `fixed_mesh_point_check.jl`, `fixed_mesh_point_meshconv.jl` — **point contacts**
  give a real signal; exact-solver validation + convergence.
- `fixed_mesh_circle.jl` — showed a plain big circle is ~0 (needs dimples).
- `fixed_mesh_show.jl` — ASCII shape dump.
