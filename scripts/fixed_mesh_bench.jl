#!/usr/bin/env julia
# scripts/fixed_mesh_bench.jl — isolate the RHS cost
#   julia --threads=auto --project=. scripts/fixed_mesh_bench.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using FermiSea, Trixi, OrdinaryDiffEq
using LinearAlgebra: norm
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.06)
sim = SimConfig(n_harmonics=4, gamma_mc=300.0, gamma_mr=0.05, gamma_3=300.0, polydeg=1)

workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel.geo"); inp = joinpath(workdir, "channel.inp")
write_channel_geo(geo, cfg)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false)

field = DensityField(cfg; nx=24, ny=16, alpha_max=100.0)

# Build a semi WITHOUT the Brinkman source (collision only) and WITH it, to
# compare rhs! cost. Reuse FermiSea plumbing directly.
equations = FermiSea.IsotropicFermiHarmonics2D(sim.n_harmonics; v_fermi=sim.v_fermi)
mesh = Trixi.P4estMesh{2}(inp; polydeg=sim.polydeg,
    boundary_symbols=[:contact_source,:contact_drain,:probe_A,:probe_B,:walls])
bcA = FermiSea.FloatingProbeBC(); bcB = FermiSea.FloatingProbeBC()
bcs = (contact_source=FermiSea.CurrentContactBC(sim.I_source),
       contact_drain=FermiSea.OhmicContactBC(0.0),
       probe_A=bcA, probe_B=bcB, walls=FermiSea.MaxwellWallBC(1.0))
collision = FermiSea.LinearCollisionMatrix(equations; gamma_mr=sim.gamma_mr,
    gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
ic0(x,t,eq) = zero(Trixi.SVector{Trixi.nvariables(eq),Float64})
solver = DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)

semi_coll = SemidiscretizationHyperbolic(mesh, equations, ic0, solver;
    boundary_conditions=bcs, source_terms=collision)
semi_brk = SemidiscretizationHyperbolic(mesh, equations, ic0, solver;
    boundary_conditions=bcs, source_terms=FermiSea.SourceTerms(collision, BrinkmanSource(field)))

ndofs = length(Trixi.semidiscretize(semi_coll, (0.0,1.0)).u0)
@printf("state length N = %d\n", ndofs); flush(stdout)

function bench(name, semi)
    ode = Trixi.semidiscretize(semi, (0.0,1.0))
    u = copy(ode.u0); du = similar(u)
    Trixi.rhs!(du, u, semi, 0.0)             # warmup/compile
    nrep = 50
    t0 = time()
    for _ in 1:nrep; Trixi.rhs!(du, u, semi, 0.0); end
    dt = (time()-t0)/nrep
    @printf("%-24s  rhs! = %.4f ms   (‖du(0)‖=%.3e)\n", name, dt*1e3, norm(du))
    flush(stdout)
    return dt
end

println("\n=== rhs! cost ==="); flush(stdout)
bench("collision only", semi_coll)
bench("collision+brinkman ρ=0", semi_brk)
paint_blob!(field, 0.5, 0.12, 0.12)
bench("collision+brinkman blob", semi_brk)

# Spectral-radius estimate (power iteration on the linearized operator via rhs!)
println("\n=== spectral radius (power iteration) ==="); flush(stdout)
ode = Trixi.semidiscretize(semi_brk, (0.0,1.0))
b = similar(ode.u0); Trixi.rhs!(b, zero(ode.u0), semi_brk, 0.0)   # affine part
scr = similar(b)
function Aop!(out, v)   # A v = rhs(v) - rhs(0)
    Trixi.rhs!(scr, v, semi_brk, 0.0); @. out = scr - b; out
end
v = randn(length(b)); v ./= norm(v); lam = 0.0
for k in 1:40
    Aop!(scr, v); lam = norm(scr); v = scr ./ lam
end
@printf("estimated spectral radius |λ|max ≈ %.1f\n", lam); flush(stdout)
@printf("=> ROCK4 stages for dt: ~sqrt(1.5*dt*|λ|). For steady, want implicit or dt-capped.\n")
println("DONE"); flush(stdout)
