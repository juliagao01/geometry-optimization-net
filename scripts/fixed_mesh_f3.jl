#!/usr/bin/env julia
# scripts/fixed_mesh_f3.jl — converge/validate f_3 (and f_4) on the optimal circle.
# (A) perturbation f_1..f_4 (recursive cascade, src perturbation_coeffs).
# (B) λ-scaling: f_2/λ, f_3/λ², f_4/λ³ should be flat (confirms Taylor orders).
# (C) INDEPENDENT cross-check: full nonlinear chord solve at several currents I, fit
#     ΔV(I) to a quartic, compare f_1..f_4 to the perturbation values.
# (D) h-convergence of f_3.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_f3.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "ogrid_geo.jl"))
using VicinityOpt, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using Gmsh: gmsh
using LinearAlgebra, Krylov, LinearMaps, Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W = 1.6, 0.8
xc,yc,R = 0.694, 0.083, 0.235          # the f_2-optimal circle
EPS = 1e-6

mesh_it(geo,inp) = (gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0);
    gmsh.open(geo); gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1);
    gmsh.model.mesh.generate(2); gmsh.write(inp); gmsh.finalize(); inp)
function make(inp, M; λ, nonlinear)
    sim=SimConfig(n_harmonics=M, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
    eqs=FermiSea.IsotropicFermiHarmonics2D(M; v_fermi=sim.v_fermi)
    mesh=Trixi.P4estMesh{2}(inp; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source,:contact_drain,:probe_A,:probe_B,:walls,:obstacle])
    bcA=FermiSea.FloatingProbeBC(); bcB=FermiSea.FloatingProbeBC()
    bcs=(contact_source=FermiSea.CurrentContactBC(1.0), contact_drain=FermiSea.OhmicContactBC(0.0),
         probe_A=bcA, probe_B=bcB, walls=FermiSea.MaxwellWallBC(1.0), obstacle=FermiSea.MaxwellWallBC(1.0))
    coll=FermiSea.LinearCollisionMatrix(eqs; gamma_mr=sim.gamma_mr, gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    src = nonlinear ? FermiSea.SourceTerms(coll, FermiSea.InertialStressSource(eqs,λ)) : FermiSea.SourceTerms(coll)
    ic0(x,t,eq)=zero(Trixi.SVector{Trixi.nvariables(eq),Float64})
    solver=DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)
    semi=SemidiscretizationHyperbolic(mesh,eqs,ic0,solver; boundary_conditions=bcs, source_terms=src)
    Trixi.semidiscretize(semi,(0.0,1.0))
    _,_,dg,cache=Trixi.mesh_equations_solver_cache(semi)
    dV(u)=(scr=similar(u); Trixi.rhs!(scr,u,semi,0.0);
           FermiSea._current_contact_potential(bcA,eqs,dg,cache)-FermiSea._current_contact_potential(bcB,eqs,dg,cache))
    (semi, dV)
end
function build(h, M; λ=1.0)
    geo=joinpath(workdir,"f3.geo"); inp=joinpath(workdir,"f3.inp")
    write_ogrid_geo(geo; L=L,W=W,xc=xc,yc=yc,R=R,ring=0.05,xPL=0.35,xPR=1.25,h=h); mesh_it(geo,inp)
    sl,dVl = make(inp,M; λ=λ, nonlinear=false)
    sn,_   = make(inp,M; λ=λ, nonlinear=true)
    (sl, sn, dVl)
end

function run()
    M = 4
    println("=== (A) perturbation f_1..f_4 (circle R=$(R), h=0.045, λ=1, M=$M) ===")
    sl,sn,dVl = build(0.045, M; λ=1.0)
    fs, info = perturbation_coeffs(sl, sn, dVl; order=4, eps=EPS)
    for n in 1:4; @printf("  f_%d = %+.6e   (GMRES it=%d conv=%s)\n", n, fs[n], info.iters[n], info.conv[n]); end
    flush(stdout)

    println("\n=== (B) λ-scaling: f_2/λ, f_3/λ², f_4/λ³ should be flat ===")
    for λ in (0.5, 1.0, 2.0)
        sl2,sn2,dVl2 = build(0.05, M; λ=λ)
        g,_ = perturbation_coeffs(sl2, sn2, dVl2; order=4, eps=EPS)
        @printf("  λ=%.2f  f1=%+.4e  f2/λ=%+.4e  f3/λ²=%+.4e  f4/λ³=%+.4e\n",
                λ, g[1], g[2]/λ, g[3]/λ^2, g[4]/λ^3); flush(stdout)
    end

    println("\n=== (C) cross-check: full nonlinear chord solve at several I, quartic fit ===")
    sl,sn,dVl = build(0.05, M; λ=1.0)
    ode=Trixi.semidiscretize(sl,(0.0,1.0)); N=length(ode.u0)
    b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),sl,0.0); scr=similar(b)
    A=LinearMap{Float64}((o,v)->(Trixi.rhs!(scr,v,sl,0.0); @. o=scr-b+EPS*v; o), N; ismutating=true)
    solveA0(r)=Krylov.gmres(A,r; restart=true,memory=100,atol=1e-11,rtol=1e-10,itmax=40000)[1]
    rnl=similar(b)
    function chord(Iamp)                       # solve A0 u + Iamp b + S_nl(u)=0
        u=solveA0(-(Iamp.*b))
        for k in 1:200
            Trixi.rhs!(rnl,u,sn,0.0); r = rnl .+ (Iamp-1.0).*b   # residual at current Iamp
            norm(r)/max(norm(u),1) < 1e-9 && break
            u .-= solveA0(r)
        end
        u
    end
    Ivals=[0.25,0.5,1.0,2.0]; dVs=Float64[]
    for I in Ivals; push!(dVs, dVl(chord(I))); @printf("  I=%.2f  ΔV=%+.6e\n", I, dVs[end]); flush(stdout); end
    Vm=hcat(Ivals,Ivals.^2,Ivals.^3,Ivals.^4); cf=Vm\dVs
    @printf("  quartic-fit:  f1=%+.4e f2=%+.4e f3=%+.4e f4=%+.4e\n", cf...)
    g,_ = perturbation_coeffs(sl, sn, dVl; order=4, eps=EPS)   # perturbation at same h
    @printf("  perturbation: f1=%+.4e f2=%+.4e f3=%+.4e f4=%+.4e\n", g...)
    @printf("  Δ(f3) fit-vs-pert = %+.2e (%.1f%%)\n", cf[3]-g[3], 100*abs(cf[3]-g[3])/max(abs(g[3]),eps())); flush(stdout)

    println("\n=== (D) h-convergence of f_3 (perturbation) ===")
    for h in (0.055, 0.045, 0.035, 0.028)
        t=time(); s1,s2,d = build(h, M; λ=1.0); gg,ii = perturbation_coeffs(s1,s2,d; order=3, eps=EPS)
        @printf("  h=%.3f N=%6d  f_1=%+.5e  f_2=%+.5e  f_3=%+.5e  (%.0fs)\n", h, ii.N, gg[1],gg[2],gg[3], time()-t)
        flush(stdout); GC.gc()
    end
end
run()
println("DONE"); flush(stdout)
