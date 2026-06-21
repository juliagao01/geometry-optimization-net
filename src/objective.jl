"""
    Objective

Glues `Geometry`, `Mesh`, and `Simulate` into one callable
`p -> -f_1(p)` for `BlackBoxOptim` (which minimizes).

Invalid geometries (obstacle leaving the channel, r(theta) collapsing, ...)
return a soft penalty proportional to how far we're outside the feasible
region so the optimizer has gradient information to push back inside.
"""
module Objective

using ..Geometry
using ..Mesh
using ..Simulate

export bounds_for, make_objective, evaluate_once, EvalState

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

"""
    bounds_for(cfg) -> Vector{Tuple{Float64,Float64}}

Default per-parameter search bounds. Order matches `unpack_params`:
    [y_c, r0, a_1, b_1, a_2, b_2, ..., a_M, b_M].
"""
function bounds_for(cfg::ChannelConfig)
    M = cfg.n_modes
    bounds = Vector{Tuple{Float64,Float64}}(undef, 2 + 2M)
    bounds[1] = (-cfg.W/4,  cfg.W/4)          # y_c
    bounds[2] = ( 0.05,    min(cfg.W/2, cfg.L_x/2) - cfg.margin)   # r0
    # a_n, b_n: amplitude shrinks with n to keep r(theta) smooth.
    for n in 1:M
        amp = 0.05 / sqrt(n)
        bounds[2 + 2n - 1] = (-amp, amp)      # a_n
        bounds[2 + 2n    ] = (-amp, amp)      # b_n
    end
    return bounds
end

# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

mutable struct EvalState
    cfg::ChannelConfig
    sim_cfg::Simulate.SimConfig
    workdir::String
    counter::Int
    history::Vector{NamedTuple}
end
EvalState(cfg, sim_cfg, workdir) = EvalState(cfg, sim_cfg, workdir, 0, NamedTuple[])

"""
    evaluate_once(p, state) -> f1

Run the full pipeline on one parameter vector, returning the *actual* f_1
(or NaN with a penalty if the geometry is rejected).
"""
function evaluate_once(p::AbstractVector{<:Real}, state::EvalState)
    state.counter += 1
    cfg = state.cfg
    o   = Geometry.unpack_params(p, cfg)

    ok, why = Geometry.validate(o, cfg)
    if !ok
        rec = (id=state.counter, status=:invalid, reason=why, f1=NaN,
               p=collect(p))
        push!(state.history, rec)
        return NaN  # caller will translate to a penalty
    end

    geo_path = joinpath(state.workdir, "shape_$(state.counter).geo")
    inp_path = joinpath(state.workdir, "shape_$(state.counter).inp")
    Geometry.write_geo(geo_path, o, cfg)
    try
        Mesh.geo_to_inp(geo_path, inp_path; verbose=false)
    catch err
        rec = (id=state.counter, status=:mesh_failed, reason=string(err),
               f1=NaN, p=collect(p))
        push!(state.history, rec)
        return NaN
    end

    f1, info = try
        Simulate.run_f1(inp_path, state.sim_cfg)
    catch err
        rec = (id=state.counter, status=:sim_failed, reason=string(err),
               f1=NaN, p=collect(p))
        push!(state.history, rec)
        return NaN
    end

    rec = (id=state.counter, status=:ok, f1=f1, info=info, p=collect(p))
    push!(state.history, rec)
    return f1
end

"""
    make_objective(cfg, sim_cfg, workdir; penalty=1.0)

Return `(p -> Float64)` suitable for `BlackBoxOptim.bboptimize`. Maximizing
f_1 = minimizing -f_1. Invalid geometries get a positive penalty.
"""
function make_objective(cfg::ChannelConfig,
                        sim_cfg::Simulate.SimConfig,
                        workdir::AbstractString;
                        penalty::Float64=1.0)
    isdir(workdir) || mkpath(workdir)
    state = EvalState(cfg, sim_cfg, workdir)
    obj(p) = begin
        f1 = evaluate_once(p, state)
        return isnan(f1) ? penalty : -f1
    end
    return obj, state
end

end # module
