#!/usr/bin/env julia
# scripts/run_optimization.jl
#
# Usage from the project root:
#     julia --project=. scripts/run_optimization.jl
#
# Stages:
#   1. Sanity check: evaluate f_1 on a hand-picked off-center circle.
#      f_1 should come out > 0 (top probe sits closer to the obstacle, so
#      density piles up there more).
#   2. Launch the optimization.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VicinityOpt
using VicinityOpt.Geometry
using VicinityOpt.Simulate
using VicinityOpt.Optimize
using VicinityOpt.Objective


# ---------------------------------------------------------------------------
# 1. Configuration. Tune these for your machine.
# ---------------------------------------------------------------------------

cfg = ChannelConfig(n_modes=0, lc=0.06, margin=0.005)   # lc stays 0.06 for FAR field
                                                         # gap region gets 0.012 from refinement
                                                         # margin: 0.005 lets us approach wall

sim_cfg = SimConfig(
    n_harmonics  = 5,
    gamma_mc     = 100.0,
    polydeg      = 3,                                    # was 2 — prof's suggestion
    t_end        = 2.0,
    residual_tol = 1e-4,
)

opt_cfg = OptConfig(
    max_evals       = 60,    # was 30
    population_size = 4,     # was 8
    workdir         = joinpath(@__DIR__, "..", "runs", "2d"),
)

# sanity check (off-center circle)
println("=== sanity check ===")
p_sanity = [0.10, 0.15]
sanity_state = VicinityOpt.Objective.EvalState(cfg, sim_cfg, opt_cfg.workdir)
mkpath(opt_cfg.workdir)
f1_sanity = VicinityOpt.Objective.evaluate_once(p_sanity, sanity_state)
@show f1_sanity

# optimize
println("\n=== optimization ===")
best_p, best_f1, state = run_optimization(cfg, sim_cfg, opt_cfg)
println("\nbest f_1 = ", best_f1)
o_best = VicinityOpt.Geometry.unpack_params(best_p, cfg)
println("y_c = ", o_best.y_c)
println("r0  = ", o_best.r0)
