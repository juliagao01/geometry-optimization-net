#!/usr/bin/env julia
# scripts/fixed_mesh_f3b.jl — remaining f_3 trust checks (A,B already passed in f3.jl):
# (D) h-convergence of f_3 (perturbation).  (C) independent cross-check: full nonlinear
# chord solve at several I, quartic fit ΔV(I) -> compare f_1..f_4 vs perturbation.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_f3b.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "ogrid_geo.jl"))
using VicinityOpt, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using Gmsh: gmsh
using LinearAlgebra, Krylov, LinearMaps, Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W = 1.6, 0.8; xc,yc,R = 0.694, 0.083, 0.235; EPS=1e-6; M=4
mesh_it(geo,inp)=(gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0); gmsh.open(geo);
    gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1); gmsh.model.mesh.generate(2); gmsh.write(inp); gmsh.finalize(); inp)
function make(inp; nonlinear)
    sim=SimConfig(n_harmonics=M, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
    eqs=FermiSea.IsotropicFermiHarmonics2D(M; v_fermi=sim.v_fermi)
    mesh=Trixi.P4estMesh{2}(inp; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source,:contact_drain,:probe_A,:probe_B,:walls,:obstacle])
    bcA=FermiSea.FloatingProbeBC(); bcB=FermiSea.FloatingProbeBC()
    bcs=(contact_source=FermiSea.CurrentContactBC(1.0), contact_drain=FermiSea.OhmicContactBC(0.0),
         probe_A=bcA, probe_B=bcB, walls=FermiSea.MaxwellWallBC(1.0), obstacle=FermiSea.MaxwellWallBC(1.0))
    coll=FermiSea.LinearCollisionMatrix(eqs; gamma_mr=sim.gamma_mr, gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    src = nonlinear ? FermiSea.SourceTerms(coll, FermiSea.InertialStressSource(eqs,1.0)) : FermiSea.SourceTerms(coll)
    ic0(x,t,eq)=zero(Trixi.SVector{Trixi.nvariables(eq),Float64})
    solver=DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)
    semi=SemidiscretizationHyperbolic(mesh,eqs,ic0,solver; boundary_conditions=bcs, source_terms=src)
    Trixi.semidiscretize(semi,(0.0,1.0)); _,_,dg,cache=Trixi.mesh_equations_solver_cache(semi)
    dV(u)=(scr=similar(u); Trixi.rhs!(scr,u,semi,0.0);
           FermiSea._current_contact_potential(bcA,eqs,dg,cache)-FermiSea._current_contact_potential(bcB,eqs,dg,cache))
    (semi,dV)
end
function build(h)
    geo=joinpath(workdir,"f3b.geo"); inp=joinpath(workdir,"f3b.inp")
    write_ogrid_geo(geo; L=L,W=W,xc=xc,yc=yc,R=R,ring=0.05,xPL=0.35,xPR=1.25,h=h); mesh_it(geo,inp)
    sl,dVl=make(inp; nonlinear=false); sn,_=make(inp; nonlinear=true); (sl,sn,dVl)
end
function run()
    println("=== (D) h-convergence of f_3 (perturbation) ===")
    for h in (0.055,0.045,0.035,0.028)
        t=time(); sl,sn,dVl=build(h); g,ii=perturbation_coeffs(sl,sn,dVl; order=3, eps=EPS)
        @printf("  h=%.3f N=%6d  f_1=%+.5e f_2=%+.5e f_3=%+.5e  (%.0fs)\n", h, ii.N, g[1],g[2],g[3], time()-t)
        flush(stdout); GC.gc()
    end
    println("\n=== (C) cross-check: nonlinear chord solve @ h=0.05, quartic fit ===")
    sl,sn,dVl=build(0.05)
    ode=Trixi.semidiscretize(sl,(0.0,1.0)); N=length(ode.u0)
    b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),sl,0.0); scr=similar(b)
    A=LinearMap{Float64}((o,v)->(Trixi.rhs!(scr,v,sl,0.0); @. o=scr-b+EPS*v; o), N; ismutating=true)
    sol(r)=Krylov.gmres(A,r; restart=true,memory=100,atol=1e-11,rtol=1e-10,itmax=40000)[1]
    rnl=similar(b)
    function chord(Iamp)
        u=sol(-(Iamp.*b))
        for k in 1:200
            Trixi.rhs!(rnl,u,sn,0.0); r=rnl .+ (Iamp-1.0).*b
            norm(r)/max(norm(u),1) < 1e-9 && break
            u .-= sol(r)
        end
        u
    end
    Ivals=[0.25,0.5,1.0,2.0]; dVs=Float64[]
    for I in Ivals; push!(dVs,dVl(chord(I))); @printf("  I=%.2f ΔV=%+.6e\n",I,dVs[end]); flush(stdout); end
    cf=hcat(Ivals,Ivals.^2,Ivals.^3,Ivals.^4)\dVs
    g,_=perturbation_coeffs(sl,sn,dVl; order=4, eps=EPS)
    @printf("  quartic-fit:  f1=%+.4e f2=%+.4e f3=%+.4e f4=%+.4e\n", cf...)
    @printf("  perturbation: f1=%+.4e f2=%+.4e f3=%+.4e f4=%+.4e\n", g...)
    @printf("  f3 agreement: fit=%+.4e pert=%+.4e  (Δ=%.1f%%)\n", cf[3], g[3], 100*abs(cf[3]-g[3])/max(abs(g[3]),eps()))
end
run(); println("DONE"); flush(stdout)
