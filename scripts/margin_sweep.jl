using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.Objective

sim_cfg = SimConfig(n_harmonics=5, gamma_mc=100.0, polydeg=2,
                    t_end=2.0, residual_tol=1e-4)

workdir = joinpath(@__DIR__, "..", "runs", "margin_sweep")
mkpath(workdir)

# For each margin m, sit at the optimum line y_c + r_0 = W/2 - m
# Pick the peak from our earlier sweep (y_c ≈ 0.05) and shift consistently
for margin in [0.02, 0.04, 0.06]
    cfg = ChannelConfig(n_modes=0, lc=0.06, margin=margin)
    state = EvalState(cfg, sim_cfg, workdir)
    target = 0.3 - margin   # y_c + r_0 = W/2 - margin
    # Match the geometric pattern of our previous peak: y_c ~ 0.05, r_0 = target - 0.05
    y_c, r0 = 0.05, target - 0.05
    println("\n=== margin = $margin  (y_c=$y_c, r_0=$r0) ===")
    f1 = evaluate_once([y_c, r0], state)
    println("f_1 = $f1")
end