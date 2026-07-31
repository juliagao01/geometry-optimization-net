module VicinityOpt

include("geometry.jl")
include("mesh.jl")
include("solver_interface.jl")
include("simulate.jl")
include("objective.jl")
include("optimize.jl")
include("fixed_mesh.jl")

using .Geometry
using .Mesh
using .SolverInterface
using .Simulate
using .Objective
using .Optimize
using .FixedMesh

# Re-export the things a user actually wants to touch from a notebook.
export ChannelConfig, ObstacleParams,
       SimConfig, OptConfig,
       unpack_params, sample_obstacle, validate,
       write_geo, geo_to_inp,
       run_f1, evaluate_once, bounds_for, make_objective,
       run_optimization,
       # fixed-mesh / density-based topology optimization
       DensityField, BrinkmanSource, write_channel_geo,
       paint_blob!, clear!, FixedEvaluator, run_f1_fixed

end # module
