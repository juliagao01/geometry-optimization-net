"""
    Optimize

Outer loop: derivative-free optimization of obstacle shape to maximize f_1.

We use BlackBoxOptim's adaptive_de_rand_1_bin_radiuslimited as the default
because it tolerates noisy / discontinuous objectives well (we have both:
mesh quality jumps, invalid geometries return penalties). CMA-ES is also
available and usually converges faster on smooth regions.
"""
module Optimize

using BlackBoxOptim
using JLD2
using ..Geometry
using ..Simulate
using ..Objective

export run_optimization, save_result

Base.@kwdef struct OptConfig
    max_evals::Int       = 200
    method::Symbol       = :adaptive_de_rand_1_bin_radiuslimited
    population_size::Int = 16
    seed::Int            = 42
    workdir::String      = joinpath(@__DIR__, "..", "runs")
    save_every::Int      = 25
end

"""
    run_optimization(cfg, sim_cfg, opt_cfg) -> (best_p, best_f1, state)

Run optimization. Periodically prints progress. Returns the best parameter
vector found, the corresponding f_1, and the full evaluation history.
"""
function run_optimization(cfg::Geometry.ChannelConfig,
                          sim_cfg::Simulate.SimConfig,
                          opt_cfg::OptConfig = OptConfig())
    isdir(opt_cfg.workdir) || mkpath(opt_cfg.workdir)
    bounds = Objective.bounds_for(cfg)
    obj, state = Objective.make_objective(cfg, sim_cfg, opt_cfg.workdir;
                                          penalty=1.0)

    println("--- vicinity_opt ---")
    println("dimensions:    ", length(bounds))
    println("budget:        ", opt_cfg.max_evals, " evaluations")
    println("workdir:       ", opt_cfg.workdir)
    println("method:        ", opt_cfg.method)

    res = bboptimize(obj;
        SearchRange   = bounds,
        Method        = opt_cfg.method,
        MaxFuncEvals  = opt_cfg.max_evals,
        PopulationSize= opt_cfg.population_size,
        RandomizeRngSeed = false,
        TraceMode     = :compact,
        TraceInterval = 5.0,
    )

    best_p  = best_candidate(res)
    best_f1 = -best_fitness(res)
    println()
    println("DONE. best f_1 = ", best_f1)
    println("best p  = ", best_p)

    save_result(opt_cfg.workdir, best_p, best_f1, state)
    return best_p, best_f1, state
end

"Persist the best parameters and history to <workdir>/result.jld2."
function save_result(workdir::AbstractString, best_p, best_f1, state)
    out = joinpath(workdir, "result.jld2")
    @save out best_p best_f1 history=state.history
    println("wrote ", out)
    return out
end

end # module
