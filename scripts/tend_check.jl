using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.Objective

workdir = joinpath(@__DIR__, "..", "runs", "tend_check"); mkpath(workdir)
cfg = ChannelConfig(n_modes=0, lc=0.06, margin=0.02)

for t_end in [2.0, 10.0, 50.0]
    sim_cfg = SimConfig(n_harmonics=5, gamma_mc=100.0, polydeg=2,
                        t_end=t_end, residual_tol=1e-4)
    state = EvalState(cfg, sim_cfg, workdir)
    println("\n=== t_end = $t_end (circle optimum) ===")
    f1 = evaluate_once([0.05, 0.23], state)
    println("f_1 = $f1")
end
