module VicinityOpt

include("geometry.jl")
include("mesh.jl")
include("simulate.jl")
include("objective.jl")
include("optimize.jl")

using .Geometry
using .Mesh
using .Simulate
using .Objective
using .Optimize

# Re-export the things a user actually wants to touch from a notebook.
export ChannelConfig, ObstacleParams,
       SimConfig, OptConfig,
       unpack_params, sample_obstacle, validate,
       write_geo, geo_to_inp,
       run_f1, evaluate_once, bounds_for, make_objective,
       run_optimization

end # module
