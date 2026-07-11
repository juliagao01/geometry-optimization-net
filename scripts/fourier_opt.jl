using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.Objective
using CMAEvolutionStrategy, JLD2

cfg = ChannelConfig(n_modes=6, lc=0.06, margin=0.02)
sim_cfg = SimConfig(n_harmonics=5, gamma_mc=100.0, polydeg=2,
                    t_end=2.0, residual_tol=1e-4)

workdir = joinpath(@__DIR__, "..", "runs", "fourier")
mkpath(workdir)

bounds = bounds_for(cfg)
lower  = Float64[b[1] for b in bounds]
upper  = Float64[b[2] for b in bounds]

# Warm start: the validated 2D circle optimum, zero Fourier coefficients
x0 = vcat([0.05, 0.23], zeros(2 * cfg.n_modes))

obj, state = make_objective(cfg, sim_cfg, workdir; penalty=1.0)

println("14D Fourier optimization, warm-started from circle optimum (f_1 = 0.385)")
result = minimize(obj, x0, 0.005;
    lower=lower, upper=upper,
    maxfevals=30,
    verbosity=1)

best_p  = xbest(result)
best_f1 = -fbest(result)
println("\nbest f_1 = ", best_f1)
o = unpack_params(best_p, cfg)
println("y_c = ", o.y_c, "  r0 = ", o.r0)
println("a_cos = ", o.a_cos)
println("b_sin = ", o.b_sin)
@save joinpath(workdir, "result.jld2") best_p best_f1 history=state.history
