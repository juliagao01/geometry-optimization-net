using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.Objective

sim_cfg = SimConfig(n_harmonics=5, gamma_mc=100.0, polydeg=2,
                    t_end=2.0, residual_tol=1e-4)
workdir = joinpath(@__DIR__, "..", "runs", "mesh_convergence_tight")
mkpath(workdir)

for lc in [0.04, 0.03]
    cfg = ChannelConfig(n_modes=0, lc=lc, margin=0.02)
    state = EvalState(cfg, sim_cfg, workdir)
    println("\n=== lc = $lc ===")
    f1 = evaluate_once([0.05, 0.23], state)
    println("f_1 = $f1")
end
