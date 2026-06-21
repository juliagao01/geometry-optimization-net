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

export SimConfig, run_f1

Base.@kwdef struct SimConfig
    n_harmonics::Int       = 10
    v_fermi::Float64       = 1.0
    gamma_mr::Float64      = 0.01
    gamma_mc::Float64      = 200.0
    gamma_3::Float64       = 200.0
    I_source::Float64      = 1.0     # any nonzero value is fine; linear model
    polydeg::Int           = 3
    t_end::Float64         = 5.0     # ample given large damping
    abstol::Float64        = 1e-7
    reltol::Float64        = 1e-7
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
        nothing,   # initial condition: zero state (linear problem, sources drive it)
        solver;
        boundary_conditions=boundary_conditions,
        source_terms=source_terms,
    )

    # Zero initial condition is fine for steady-state of a linear damped system.
    tspan = (0.0, sim.t_end)
    ode = Trixi.semidiscretize(semi, tspan)

    steady_state = SteadyStateCallback(; abstol=sim.residual_tol,
                                         reltol=sim.residual_tol)
    summary_cb   = sim.verbose ? SummaryCallback() : nothing
    stepsize_cb  = StepsizeCallback(cfl=sim.cfl)
    callbacks    = CallbackSet(filter(!isnothing,
                                      (steady_state, stepsize_cb, summary_cb))...)

    sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false);
        dt=1.0e-3, save_everystep=false,
        callback=callbacks)

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

end # module
