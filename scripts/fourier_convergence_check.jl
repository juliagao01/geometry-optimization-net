using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.Objective

sim_cfg = SimConfig(n_harmonics=5, gamma_mc=100.0, polydeg=2,
                    t_end=2.0, residual_tol=1e-4)
workdir = joinpath(@__DIR__, "..", "runs", "fourier_convergence")
mkpath(workdir)

p = [0.04934956100663416, 0.21966370021431364,
     0.0018460171136576997, 0.0018490975593675948,
     -0.002676311968158894, -0.01087452106620217,
     -0.00031421454946573, -0.0007703225607678249,
     -0.009407357442259555, -0.012293162109548167,
     -0.0065566518945101085, -0.0009289067364181385,
     -0.008722133172310199, -0.006425422433685485]

for lc in [0.06, 0.04]
    cfg = ChannelConfig(n_modes=6, lc=lc, margin=0.02)
    state = EvalState(cfg, sim_cfg, workdir)
    println("\n=== lc = $lc ===")
    f1 = evaluate_once(p, state)
    println("f_1 = $f1")
end
