"""
    Simulate

Run one FermiSea.jl steady-state simulation for a given mesh and return
the linear-response coefficient

    f_1 = (V_A - V_B) / I

where V_A, V_B are the self-consistent floating-probe potentials assembled
by `FloatingProbeBC`. Because IsotropicFermiHarmonics2D is linear, `f_1` is
independent of `I` (we just pick one).
"""
module Simulate

using FermiSea
using Trixi
using OrdinaryDiffEq
using ..SolverInterface
using Krylov
using LinearMaps
using LinearAlgebra: norm

# Linear damped system: zero initial state is fine; sources drive it to steady state.
@inline initial_condition_zero(
    x, t, equations::FermiSea.IsotropicFermiHarmonics2D{NVARS}
) where {NVARS} = zero(Trixi.SVector{NVARS, Float64})

export SimConfig, run_f1, run_f1_steady

Base.@kwdef struct SimConfig <: SolverInterface.AbstractSolverConfig
    n_harmonics::Int       = 4       # was 10; his suggestion (m_max=4)
    gamma_mr::Float64      = 0.05    # was 0.01; stabilizes
    gamma_mc::Float64      = 300.0   # was 200; deeper hydrodynamic
    gamma_3::Float64       = 300.0   # match gamma_mc
    I_source::Float64      = 1.0     # any nonzero value is fine; linear model
    polydeg::Int           = 3
    t_end::Float64         = 2000.0     # ample given large damping
    abstol::Float64        = 1e-7
    reltol::Float64        = 1e-7
    v_fermi::Float64       = 1.0
    cfl::Float64           = 0.5
    residual_tol::Float64  = 1e-6    # steady-state stopping
    verbose::Bool          = false
end

"""
    run_f1(inp_path, sim::SimConfig) -> (f1, info)

Solve the linear Boltzmann moment problem on the given mesh and return
`f1 = (V_A - V_B) / I`. The second element is a small NamedTuple with
diagnostics (V_A, V_B, current_through_source, walltime, steps).
"""
function run_f1(inp_path::AbstractString, sim::SimConfig=SimConfig())
    t0 = time()

    equations = IsotropicFermiHarmonics2D(sim.n_harmonics; v_fermi=sim.v_fermi)

    mesh = Trixi.P4estMesh{2}(inp_path; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source, :contact_drain,
                          :probe_A, :probe_B, :walls])

    bc_source  = FermiSea.CurrentContactBC(sim.I_source)
    bc_drain   = FermiSea.OhmicContactBC(0.0)
    bc_probe_A = FermiSea.FloatingProbeBC()
    bc_probe_B = FermiSea.FloatingProbeBC()
    bc_walls   = FermiSea.MaxwellWallBC(1.0)

    # Trixi expects a NamedTuple keyed by the same Symbols passed as
    # boundary_symbols when the mesh was loaded.
    boundary_conditions = (
        contact_source = bc_source,
        contact_drain  = bc_drain,
        probe_A        = bc_probe_A,
        probe_B        = bc_probe_B,
        walls          = bc_walls,
    )

    collision = FermiSea.LinearCollisionMatrix(equations;
        gamma_mr=sim.gamma_mr, gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    source_terms = FermiSea.SourceTerms(collision)

    solver = DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)

    semi = SemidiscretizationHyperbolic(mesh, equations,
        initial_condition_zero,  # initial condition: zero state (linear problem, sources drive it)
        solver;
        boundary_conditions=boundary_conditions,
        source_terms=source_terms,
    )

    # Zero initial condition is fine for steady-state of a linear damped system.
    tspan = (0.0, sim.t_end)
    ode = Trixi.semidiscretize(semi, tspan)

    steady_state = SteadyStateCallback(; abstol=1e-4, reltol=0.0)
    summary_cb   = sim.verbose ? SummaryCallback() : nothing
    callbacks    = CallbackSet(filter(!isnothing,
                                      (steady_state, summary_cb))...)

    sol = solve(ode, ROCK4();
        save_everystep=false,
        callback=callbacks,
        abstol=1e-5, reltol=1e-5)

    # Push the final u through the RHS once to force boundary state update,
    # then read off the floating probe potentials. Without this step, the
    # cache.boundaries.u that _current_contact_potential reads from may
    # still reflect an earlier time level.
    u_final = sol.u[end]
    du = similar(u_final)
    Trixi.rhs!(du, u_final, semi, sol.t[end])

    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    V_A = FermiSea._current_contact_potential(bc_probe_A, equations, dg, cache)
    V_B = FermiSea._current_contact_potential(bc_probe_B, equations, dg, cache)
    f1  = (V_A - V_B) / sim.I_source

    info = (V_A = V_A, V_B = V_B,
            I_source = sim.I_source,
            walltime = time() - t0, steps = length(sol.t))
    return f1, info
end


"""
    run_f1_steady(inp_path, sim::SimConfig) -> (f1, info)

Solve the *steady-state* problem directly as a linear system, instead of
time-stepping. Since IsotropicFermiHarmonics2D is linear, the steady state
satisfies rhs(u) = 0, i.e. A*u = -b where rhs(u) = A*u + b. We solve this
matrix-free with GMRES, using rhs! evaluations as the matvec.

This is exact (no transient contamination) and typically 10-100x faster
than time-stepping through slow viscous modes.
"""
function run_f1_steady(inp_path::AbstractString, sim::SimConfig=SimConfig())
    t0 = time()

    equations = IsotropicFermiHarmonics2D(sim.n_harmonics; v_fermi=sim.v_fermi)

    mesh = Trixi.P4estMesh{2}(inp_path; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source, :contact_drain,
                          :probe_A, :probe_B, :walls])

    bc_source  = FermiSea.CurrentContactBC(sim.I_source)
    bc_drain   = FermiSea.OhmicContactBC(0.0)
    bc_probe_A = FermiSea.FloatingProbeBC()
    bc_probe_B = FermiSea.FloatingProbeBC()
    bc_walls   = FermiSea.MaxwellWallBC(1.0)

    boundary_conditions = (
        contact_source = bc_source,
        contact_drain  = bc_drain,
        probe_A        = bc_probe_A,
        probe_B        = bc_probe_B,
        walls          = bc_walls,
    )

    collision = FermiSea.LinearCollisionMatrix(equations;
        gamma_mr=sim.gamma_mr, gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    source_terms = FermiSea.SourceTerms(collision)

    solver = DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)

    semi = SemidiscretizationHyperbolic(mesh, equations,
        initial_condition_zero, solver;
        boundary_conditions=boundary_conditions,
        source_terms=source_terms)

    # Get a template state vector and the affine part b = rhs(0)
    ode = Trixi.semidiscretize(semi, (0.0, 1.0))
    u0 = copy(ode.u0)            # zero vector (initial_condition_zero)
    fill!(u0, 0.0)
    N = length(u0)

    b = similar(u0)
    Trixi.rhs!(b, u0, semi, 0.0)  # rhs at u=0 gives the affine forcing b

    # Matrix-free A*v = rhs(v) - b  (linearity)
    scratch = similar(u0)
    function matvec!(out, v)
        Trixi.rhs!(scratch, v, semi, 0.0)
        @. out = scratch - b
        return out
    end
    A = LinearMap{Float64}((out, v) -> matvec!(out, v), N; ismutating=true)

    # Solve A*u = -b
    rhs_vec = -b
    u_steady, stats = Krylov.bicgstab(A, rhs_vec;
        atol=sim.residual_tol, rtol=sim.residual_tol,
        itmax=1000, verbose=0)

    converged = stats.solved
    # Final rhs! call to sync boundary caches (floating probe potentials)
    du = similar(u_steady)
    Trixi.rhs!(du, u_steady, semi, 0.0)
    residual = norm(du) / max(norm(u_steady), 1.0)

    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    V_A = FermiSea._current_contact_potential(bc_probe_A, equations, dg, cache)
    V_B = FermiSea._current_contact_potential(bc_probe_B, equations, dg, cache)
    f1  = (V_A - V_B) / sim.I_source

    info = (V_A=V_A, V_B=V_B, I_source=sim.I_source,
            walltime=time() - t0, gmres_iters=stats.niter,
            converged=converged, residual=residual)
    return f1, info
end

# NOTE: run_f1_steady (direct GMRES/BiCGStab solve) does NOT converge on this
# operator — needs preconditioning or in-FermiSea implementation. Do not use
# until fixed. Time-stepping requires t_end >~ 100 for gamma_mc=100 (slow but correct).
function SolverInterface.evaluate(cfg::SimConfig, inp_path::AbstractString)
    f1, info = run_f1(inp_path, cfg)
    return (objective = f1, walltime = info.walltime, V_A = info.V_A, V_B = info.V_B)
end

end # module
