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

cfg = ChannelConfig(
    L_x       = 1.0,
    W         = 0.6,
    x_probe   = 0.5,
    L_probe   = 0.15,
    x_c       = 0.5,
    n_modes   = 6,
    lc        = 0.04,
)

sim_cfg = SimConfig(
    n_harmonics  = 10,         # m_max in the user's notation
    v_fermi      = 1.0,
    gamma_mr     = 0.01,
    gamma_mc     = 200.0,      # deep hydrodynamic regime, as requested
    gamma_3      = 200.0,
    I_source     = 1.0,
    polydeg      = 3,
    t_end        = 5.0,
    residual_tol = 1e-6,
    verbose      = false,
)

opt_cfg = OptConfig(
    max_evals       = 200,
    method          = :adaptive_de_rand_1_bin_radiuslimited,
    population_size = 16,
    seed            = 42,
    workdir         = joinpath(@__DIR__, "..", "runs", "main"),
)

# ---------------------------------------------------------------------------
# 2. Sanity check on one shape: off-center circle (y_c = 0.1, r0 = 0.15).
# ---------------------------------------------------------------------------

println("=== sanity check ===")
M = cfg.n_modes
p_sanity = vcat([0.10, 0.15], zeros(2M))
sanity_state = VicinityOpt.Objective.EvalState(cfg, sim_cfg, opt_cfg.workdir)
mkpath(opt_cfg.workdir)
f1_sanity = VicinityOpt.Objective.evaluate_once(p_sanity, sanity_state)
@show f1_sanity
@show sanity_state.history[end]

# ---------------------------------------------------------------------------
# 3. Real optimization run.
# ---------------------------------------------------------------------------

println("\n=== optimization ===")
best_p, best_f1, state = run_optimization(cfg, sim_cfg, opt_cfg)
println("\nbest f_1 = ", best_f1)
o_best = VicinityOpt.Geometry.unpack_params(best_p, cfg)
println("best obstacle: y_c=", o_best.y_c, "  r0=", o_best.r0)
println("a_cos: ", o_best.a_cos)
println("b_sin: ", o_best.b_sin)
