#!/usr/bin/env julia
# scripts/fixed_mesh_steady.jl — can we solve the steady state DIRECTLY (fast)?
#   julia --threads=auto --project=. scripts/fixed_mesh_steady.jl
#
# The problem is LINEAR: steady state solves rhs(u)=0 i.e. A u = -b, with
# A v := rhs(v) - rhs(0) (matrix-free via rhs!). Explicit time-stepping is
# hopeless here (stiff + slow diffusive mode). We test matrix-free GMRES,
# unpreconditioned vs a cheap diagonal preconditioner built from the known
# per-harmonic collision rates (the undamped density mode gets a streaming/LLF
# estimate so it isn't left at 0). Goal: steady f_1 in seconds.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using FermiSea, Trixi
using Krylov, LinearMaps
using LinearAlgebra: norm, Diagonal
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.06)
sim = SimConfig(n_harmonics=4, gamma_mc=300.0, gamma_mr=0.05, gamma_3=300.0, polydeg=1)

workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel.geo"); inp = joinpath(workdir, "channel.inp")
write_channel_geo(geo, cfg)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false)

field = DensityField(cfg; nx=24, ny=16, alpha_max=100.0)
paint_blob!(field, 0.5, 0.12, 0.12)   # off-center: expect f_1 > 0

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
semi = SemidiscretizationHyperbolic(mesh, equations, ic0, solver;
    boundary_conditions=bcs, source_terms=FermiSea.SourceTerms(collision, BrinkmanSource(field)))

ode = Trixi.semidiscretize(semi, (0.0,1.0))
N = length(ode.u0)
nvars = Trixi.nvariables(equations)

b = similar(ode.u0); Trixi.rhs!(b, zero(ode.u0), semi, 0.0)   # affine forcing
scr = similar(b)
matvec!(out, v) = (Trixi.rhs!(scr, v, semi, 0.0); @. out = scr - b; out)
A = LinearMap{Float64}((out,v)->matvec!(out,v), N; ismutating=true)
rhs_vec = -b

# --- diagonal preconditioner from known rates ---
rates = [collision.W[i,i] for i in 1:nvars]     # 0, gmr, gmr, gmr+gmc, ...
dx = cfg.lc
vdx = sim.v_fermi * (2*sim.polydeg + 1) / dx     # streaming/LLF diagonal estimate
diagest = [rates[i] + (i==1 ? vdx : 0.0) + (i>1 ? field.alpha_max*0.0 : 0.0) for i in 1:nvars]
# M ≈ A^{-1} (diagonal). A's diagonal ≈ -(rate + vdx-ish); make M negative so M*A ≈ +I.
pvec = [-1.0/diagest[i] for i in 1:nvars]
precond_full = repeat(pvec, N ÷ nvars)
Mmap = LinearMap{Float64}((out,v)->(@. out = v * precond_full), N; ismutating=true)

function read_f1()
    # A final rhs! at the current solution syncs boundary caches
    _,_,dg,cache = Trixi.mesh_equations_solver_cache(semi)
    V_A = FermiSea._current_contact_potential(bcA, equations, dg, cache)
    V_B = FermiSea._current_contact_potential(bcB, equations, dg, cache)
    (V_A - V_B)/sim.I_source, V_A, V_B
end

function trial(name; kwargs...)
    t0 = time()
    u, stats = Krylov.gmres(A, rhs_vec; kwargs...)
    du = similar(u); Trixi.rhs!(du, u, semi, 0.0); res = norm(du)/max(norm(u),1)
    f1,VA,VB = read_f1()
    @printf("%-28s f_1=%+.4e  solved=%s iters=%d res=%.2e  %.1fs\n",
            name, f1, stats.solved, stats.niter, res, time()-t0)
    flush(stdout)
    return u, stats
end

function trial_bicg(name; kwargs...)
    t0 = time()
    u, stats = Krylov.bicgstab(A, rhs_vec; kwargs...)
    du = similar(u); Trixi.rhs!(du, u, semi, 0.0); res = norm(du)/max(norm(u),1)
    f1,_,_ = read_f1()
    @printf("%-28s f_1=%+.4e  solved=%s iters=%d res=%.2e  %.1fs\n",
            name, f1, stats.solved, stats.niter, res, time()-t0)
    flush(stdout)
    return u, stats
end

println("N=$N  rates=$(round.(rates,digits=3))"); flush(stdout)

println("\n=== f_1 convergence vs GMRES iterations (unpreconditioned) ==="); flush(stdout)
for it in (500, 1000, 1500, 2000, 3000)
    trial("gmres mem=50 itmax=$it"; restart=true, memory=50, atol=0.0, rtol=0.0, itmax=it)
end

println("\n=== BiCGStab (short recurrence, cheaper/iter) ==="); flush(stdout)
trial_bicg("bicgstab itmax=1000"; atol=1e-7, rtol=1e-7, itmax=1000)
trial_bicg("bicgstab itmax=2000"; atol=1e-8, rtol=1e-8, itmax=2000)
println("DONE"); flush(stdout)
