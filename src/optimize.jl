"""
    Optimize

Outer loop: derivative-free optimization of obstacle shape to maximize f_1.
Uses Optim.jl's ParticleSwarm — same family as the BBO default
(population-based, no gradients), but actively maintained.
"""
module Optimize

using Optim
using JLD2
using ..Geometry
using ..Simulate
using ..Objective

export OptConfig, run_optimization, save_result

Base.@kwdef struct OptConfig
    max_evals::Int       = 200
    population_size::Int = 16
    seed::Int            = 42
    workdir::String      = joinpath(@__DIR__, "..", "runs")
end

"""
    run_optimization(cfg, sim_cfg, opt_cfg) -> (best_p, best_f1, state)
"""
function run_optimization(cfg::Geometry.ChannelConfig,
                          sim_cfg::Simulate.SimConfig,
                          opt_cfg::OptConfig = OptConfig())
    isdir(opt_cfg.workdir) || mkpath(opt_cfg.workdir)

    bounds = Objective.bounds_for(cfg)
    lower  = Float64[b[1] for b in bounds]
    upper  = Float64[b[2] for b in bounds]
    x0     = (lower .+ upper) ./ 2  # center of the box

    obj, state = Objective.make_objective(cfg, sim_cfg, opt_cfg.workdir;
                                          penalty=1.0)

    println("--- vicinity_opt ---")
    println("dimensions: ", length(bounds))
    println("budget:     ", opt_cfg.max_evals, " function evaluations")
    println("workdir:    ", opt_cfg.workdir)
    println("method:     ParticleSwarm (Optim.jl)")
    println()

    # ParticleSwarm: iterations = max generations; total evals ~ iterations * n_particles.
    # We bound by max_evals via Optim.Options.iterations.
    n_particles = opt_cfg.population_size
    max_iters   = max(1, opt_cfg.max_evals ÷ n_particles)

    result = Optim.optimize(
        obj, lower, upper, x0,
        ParticleSwarm(lower=lower, upper=upper, n_particles=n_particles),
        Optim.Options(iterations=max_iters,
                      show_trace=true,
                      show_every=1,
                      f_calls_limit=opt_cfg.max_evals),
    )

    best_p  = Optim.minimizer(result)
    best_f1 = -Optim.minimum(result)
    println()
    println("DONE. best f_1 = ", best_f1)
    println("best p  = ", best_p)

    save_result(opt_cfg.workdir, best_p, best_f1, state)
    return best_p, best_f1, state
end

function save_result(workdir::AbstractString, best_p, best_f1, state)
    out = joinpath(workdir, "result.jld2")
    @save out best_p best_f1 history=state.history
    println("wrote ", out)
    return out
end

end # module