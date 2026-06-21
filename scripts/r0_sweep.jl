using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.Objective
using JLD2

cfg = ChannelConfig(n_modes=0, lc=0.06, margin=0.02)
sim_cfg = SimConfig(n_harmonics=5, gamma_mc=100.0, polydeg=2,
                    t_end=2.0, residual_tol=1e-4)

workdir = joinpath(@__DIR__, "..", "runs", "r0_sweep")
mkpath(workdir)

# Sweep r_0 along the line y_c + r_0 = 0.27 (gap = 0.03 from top wall)
# Constraint: y_c must be in [-0.15, 0.15], so r_0 must be > 0.12.
configs = [
    (0.15, 0.12),   # y_c at upper bound, small r0
    (0.12, 0.15),
    (0.09, 0.18),
    (0.06, 0.21),
    (0.03, 0.24),
    (0.00, 0.27),   # centered, biggest r0
]

state = EvalState(cfg, sim_cfg, workdir)
println("=== r_0 sweep along y_c + r_0 = 0.27 ===")
for (y_c, r0) in configs
    f1 = evaluate_once([y_c, r0], state)
    println("y_c=$y_c, r_0=$r0  ->  f_1 = $f1")
end
@save joinpath(workdir, "result.jld2") history=state.history