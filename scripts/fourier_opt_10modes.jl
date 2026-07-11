using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.Objective
using CMAEvolutionStrategy, JLD2

cfg = ChannelConfig(n_modes=10, lc=0.06, margin=0.02)
sim_cfg = SimConfig(n_harmonics=5, gamma_mc=100.0, polydeg=2,
                    t_end=2.0, residual_tol=1e-4)

workdir = joinpath(@__DIR__, "..", "runs", "fourier10")
mkpath(workdir)

bounds = bounds_for(cfg)
lower  = Float64[b[1] for b in bounds]
upper  = Float64[b[2] for b in bounds]

prev_best = [0.04934956100663416, 0.21966370021431364,
     0.0018460171136576997, 0.0018490975593675948,
     -0.002676311968158894, -0.01087452106620217,
     -0.00031421454946573, -0.0007703225607678249,
     -0.009407357442259555, -0.012293162109548167,
     -0.0065566518945101085, -0.0009289067364181385,
     -0.008722133172310199, -0.006425422433685485]
x0 = vcat(prev_best, zeros(8))

obj, state = make_objective(cfg, sim_cfg, workdir; penalty=1.0)

println("14D->22D Fourier optimization, warm-started from 6-mode best (f_1=0.972)")
result = minimize(obj, x0, 0.005;
    lower=lower, upper=upper,
    maxfevals=60,
    verbosity=1)

best_p  = xbest(result)
best_f1 = -fbest(result)
println("\nbest f_1 = ", best_f1)
o = unpack_params(best_p, cfg)
println("y_c = ", o.y_c, "  r0 = ", o.r0)
println("a_cos = ", o.a_cos)
println("b_sin = ", o.b_sin)
@save joinpath(workdir, "result.jld2") best_p best_f1 history=state.history
