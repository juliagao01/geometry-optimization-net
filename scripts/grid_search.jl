using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.Objective
using JLD2

cfg = ChannelConfig(n_modes=0, lc=0.06, margin=0.015)
sim_cfg = SimConfig(
    n_harmonics  = 5,
    gamma_mc     = 100.0,
    polydeg      = 2,        # 3????
    t_end        = 2.0,
    residual_tol = 1e-4,
)

workdir = joinpath(@__DIR__, "..", "runs", "constraint_sweep")
mkpath(workdir)


configs = [
    (0.030, 0.254),
    (0.050, 0.234),
    (0.070, 0.214),
    (0.090, 0.194),
    (0.110, 0.174),
]

state = EvalState(cfg, sim_cfg, workdir)
results = []
for (y_c, r0) in configs
    println("\n=== y_c=$y_c, r0=$r0 ===")
    f1 = evaluate_once([y_c, r0], state)
    push!(results, (y_c=y_c, r0=r0, f1=f1))
    @show f1
end

println("\n=== summary ===")
for r in results
    println("y_c=$(r.y_c), r0=$(r.r0)  ->  f_1=$(r.f1)")
end
imax = argmax([r.f1 for r in results])
println("\npeak: $(results[imax])")
@save joinpath(workdir, "result.jld2") results