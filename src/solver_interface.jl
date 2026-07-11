"""
    SolverInterface

Defines the contract every solver backend must satisfy so that Geometry,
Mesh, Objective, and Optimize stay solver-agnostic. To add a new solver
(nonlinear Boltzmann, Navier-Stokes, etc.), implement `evaluate(::YourSolverConfig, inp_path)`
returning a NamedTuple with at least a `objective` field.
"""
module SolverInterface

export AbstractSolverConfig, evaluate

abstract type AbstractSolverConfig end

"""
    evaluate(config::AbstractSolverConfig, inp_path::AbstractString) -> NamedTuple

Run one simulation on the given mesh and return a NamedTuple with at least:
  - `objective::Float64`  the scalar quantity to optimize (e.g. f_1)
  - `walltime::Float64`
Every concrete solver config (SimConfig, future NonlinearSimConfig, ...)
must add a method for its own type.
"""
function evaluate end

end # module
